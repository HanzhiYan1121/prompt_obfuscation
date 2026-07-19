#!/usr/bin/env python3
"""End-to-end InferDPT adaptation for strict black-box prompt recovery."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "recovery_tools"))

from emojiprompt_e2e.run_emojiprompt_e2e import (  # noqa: E402
    CATEGORIES,
    DEFAULT_STRUCTURED_KEYS,
    atomic_write_json,
    atomic_write_text,
    audit_recovery_inputs,
    build_blackbox_recovery_prompt,
    cleanup_cuda,
    collect_stage_records,
    generate_plain_outputs,
    read_git_commit,
    reconstruct_prompt,
    resolve_output,
    sample_dir,
    shard_items,
    stage_recover,
)
from recovery_tools.run_strict_blackbox_step2 import (  # noqa: E402
    load_json,
    load_test_queries,
    mean,
    parse_structured_keys,
    result_path,
    token_f1,
)
from src.model import Model  # noqa: E402
from src.utils import set_seed  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+|[^\w\s]", re.UNICODE)
ALPHA_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_'-]*$")

COMMON_WORDS = """
assistant answer user task format concise detailed explain analyze classify summarize rewrite
extract identify generate continue follow avoid include exclude JSON markdown table code tool
system prompt instruction role context constraint policy safety output input question response
step reason evidence data document private public secure privacy token text language model
helpful professional accurate brief friendly strict only never always when if then use return
""".split()


def gen_args(args: argparse.Namespace, max_new_tokens: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        victim_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens if max_new_tokens is None else max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def original_prompts(manifest: dict) -> list[str]:
    out = []
    for item in manifest["sample"]:
        out.append((result_path(item) / "original_prompt.txt").read_text(encoding="utf-8").strip())
    return out


class LlamaLexicalPerturber:
    def __init__(self, model: Model, manifest: dict, args: argparse.Namespace):
        self.model = model
        self.args = args
        self.rng = random.Random(args.seed)
        vocab = set(COMMON_WORDS)
        for prompt in original_prompts(manifest):
            for tok in WORD_RE.findall(prompt):
                if ALPHA_RE.match(tok) and 2 <= len(tok) <= 24:
                    vocab.add(tok.lower())
        self.vocab = sorted(vocab)
        self.embeddings = self._embed_words(self.vocab)
        if self.embeddings.shape[0] == 0:
            raise RuntimeError("No lexical embeddings could be built for InferDPT perturbation.")
        self.delta = np.maximum(self.embeddings.std(axis=0) * 2.0, 1e-4)

    def _embed_words(self, words: list[str]) -> np.ndarray:
        emb_layer = self.model.word_embedding_layer
        rows = []
        kept = []
        for word in words:
            ids = self.model.tokenizer(word, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if ids.numel() == 0 or ids.numel() > self.args.max_subtokens_per_word:
                continue
            with torch.no_grad():
                vec = emb_layer(ids.to(emb_layer.weight.device)).float().mean(dim=0).detach().cpu().numpy()
            rows.append(vec)
            kept.append(word)
        self.vocab = kept
        return np.stack(rows).astype(np.float32)

    def perturb_word(self, word: str, rng: random.Random) -> str:
        clean = word.lower()
        if clean.isnumeric():
            return str(rng.randint(1, 1000))
        if not ALPHA_RE.match(clean) or len(clean) <= 2:
            return word
        ids = self.model.tokenizer(clean, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if ids.numel() == 0 or ids.numel() > self.args.max_subtokens_per_word:
            return word if rng.random() < self.args.keep_oov_probability else ""
        emb_layer = self.model.word_embedding_layer
        with torch.no_grad():
            origin = emb_layer(ids.to(emb_layer.weight.device)).float().mean(dim=0).detach().cpu().numpy()
        beta = self.delta / max(self.args.epsilon, 1e-6)
        noisy = origin + np.asarray([rng.expovariate(1.0 / b) * (1 if rng.random() < 0.5 else -1) for b in beta], dtype=np.float32)
        dists = np.linalg.norm(self.embeddings - noisy[None, :], axis=1)
        k = min(self.args.adjacency_size, len(self.vocab))
        idx = np.argpartition(dists, k - 1)[:k]
        local_dists = dists[idx]
        scale = max(float(local_dists.max()), 1e-6)
        logits = self.args.epsilon * (scale - local_dists) / (2.0 * scale)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        chosen = rng.choices(list(idx), weights=probs.tolist(), k=1)[0]
        return self.vocab[int(chosen)]

    def perturb_prompt(self, prompt: str, sample_index: int) -> tuple[str, list[dict]]:
        rng = random.Random(f"{self.args.seed}:inferdpt:{sample_index}")
        toks = WORD_RE.findall(prompt)
        if self.args.max_prompt_tokens > 0:
            toks = toks[: self.args.max_prompt_tokens]
        out = []
        trace = []
        for tok in toks:
            if tok in "{}[]()<>/:._-|":
                out.append(tok)
                continue
            new_tok = self.perturb_word(tok, rng)
            if new_tok:
                out.append(new_tok)
            if tok != new_tok:
                trace.append({"raw": tok, "perturbed": new_tok})
        return detokenize(out), trace


def remote_system_prompt(perturbed_prompt: str) -> str:
    return (
        "You are a remote closed-box LLM inside an InferDPT-style privacy-preserving pipeline. "
        "The following system instruction has been token-level perturbed by local differential privacy. "
        "Infer its likely intent and answer the user as well as possible.\n\n"
        f"Perturbed system instruction:\n{perturbed_prompt}"
    )


def extraction_system_prompt() -> str:
    return (
        "You are the trusted local extraction module of InferDPT. Use the raw private system "
        "instruction and the remote perturbed generation to reconstruct the final answer. "
        "Return only the final answer to the user."
    )


def extraction_user_prompt(raw_prompt: str, user_query: str, perturbed_generation: str) -> str:
    return (
        "Raw private system instruction:\n"
        f"{raw_prompt}\n\n"
        "User query:\n"
        f"{user_query}\n\n"
        "Perturbed generation from remote LLM:\n"
        f"{perturbed_generation}\n\n"
        "Final answer:"
    )


def generate_inferdpt_outputs(model: Model, raw_prompt: str, perturbed_prompt: str, queries: list[str], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    remote_outputs = generate_plain_outputs(model, remote_system_prompt(perturbed_prompt), queries, gen_args(args))
    extractor_inputs = [extraction_user_prompt(raw_prompt, q, y) for q, y in zip(queries, remote_outputs)]
    final_outputs = generate_plain_outputs(model, extraction_system_prompt(), extractor_inputs, gen_args(args))
    return final_outputs, remote_outputs


def load_source_manifest(args: argparse.Namespace) -> dict:
    path = args.source_manifest if args.source_manifest.is_absolute() else ROOT / args.source_manifest
    return load_json(path)


def stage_manifest(args: argparse.Namespace) -> dict:
    out_dir = resolve_output(args.output_dir)
    path = out_dir / "manifest.json"
    if path.exists() and not args.overwrite_manifest:
        return load_json(path)
    source = load_source_manifest(args)
    payload = {
        "setting": "inferdpt_llama31_strict_blackbox",
        "description": "InferDPT open-source adaptation for hidden system prompts.",
        "source_manifest": str(args.source_manifest),
        "official_repo": {
            "path": "third_party/InferDPT",
            "commit": read_git_commit(ROOT / "third_party" / "InferDPT"),
            "note": "Official code requires external cl100_embeddings.json; this run uses local Llama lexical embeddings.",
        },
        "victim_model": args.victim_model,
        "recovery_model": args.recovery_model,
        "recovery_result_prefix": args.result_prefix,
        "recovery_model_tag": args.recovery_model_tag,
        "n_prompts": min(args.n_prompts, len(source["sample"])),
        "evidence_queries": source["evidence_queries"],
        "selection_queries": source["selection_queries"],
        "eval_queries": source["eval_queries"],
        "epsilon": args.epsilon,
        "adjacency_size": args.adjacency_size,
        "sample": source["sample"][: args.n_prompts],
    }
    atomic_write_json(path, payload)
    return payload


def stage_perturb(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return
    model = Model(args.victim_model, "4bit" if args.quantize_4bit else None)
    perturber = LlamaLexicalPerturber(model, manifest, args)
    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "perturbation.json"
        if path.exists() and not args.overwrite:
            print(f"skip perturbation {item['sample_index']:03d} {item['id']}", flush=True)
            continue
        raw_prompt = (result_path(item) / "original_prompt.txt").read_text(encoding="utf-8").strip()
        perturbed_prompt, trace = perturber.perturb_prompt(raw_prompt, item["sample_index"])
        payload = {
            "sample_index": item["sample_index"],
            "id": item["id"],
            "original_prompt": raw_prompt,
            "perturbed_prompt": perturbed_prompt,
            "epsilon": args.epsilon,
            "adjacency_size": args.adjacency_size,
            "replacement_trace": trace,
            "text_f1_perturbed_vs_original": token_f1(perturbed_prompt, raw_prompt),
        }
        atomic_write_json(path, payload)
        atomic_write_text(sample_dir(out_dir, item["sample_index"]) / "perturbed_prompt.txt", perturbed_prompt)
        print(f"perturbed sample={item['sample_index']:03d} replacements={len(trace)}", flush=True)
    cleanup_cuda(model, perturber)


def stage_collect(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return
    set_seed(args.seed + args.shard_index)
    model = Model(args.victim_model, "4bit" if args.quantize_4bit else None)
    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "transcript.json"
        if path.exists() and not args.overwrite:
            print(f"skip transcript {item['sample_index']:03d} {item['id']}", flush=True)
            continue
        ppath = sample_dir(out_dir, item["sample_index"]) / "perturbation.json"
        if not ppath.exists():
            raise FileNotFoundError(f"Missing perturbation: {ppath}")
        perturb = load_json(ppath)
        all_queries = load_test_queries(
            result_path(item),
            manifest["evidence_queries"] + manifest["selection_queries"] + manifest["eval_queries"],
        )
        evidence = all_queries[: manifest["evidence_queries"]]
        selection = all_queries[manifest["evidence_queries"] : manifest["evidence_queries"] + manifest["selection_queries"]]
        eval_queries = all_queries[
            manifest["evidence_queries"] + manifest["selection_queries"] :
            manifest["evidence_queries"] + manifest["selection_queries"] + manifest["eval_queries"]
        ]
        print(f"collect sample={item['sample_index']:03d} {item['id']}", flush=True)
        evidence_outputs, evidence_remote = generate_inferdpt_outputs(model, perturb["original_prompt"], perturb["perturbed_prompt"], evidence, args)
        selection_outputs, selection_remote = generate_inferdpt_outputs(model, perturb["original_prompt"], perturb["perturbed_prompt"], selection, args) if selection else ([], [])
        payload = {
            "sample_index": item["sample_index"],
            "id": item["id"],
            "evidence": [{"user_input": q, "protected_output": o} for q, o in zip(evidence, evidence_outputs)],
            "selection": [{"user_input": q, "protected_output": o} for q, o in zip(selection, selection_outputs)],
            "eval_inputs": eval_queries,
            "internal_remote_outputs_not_used_by_attacker": {
                "evidence": evidence_remote,
                "selection": selection_remote,
            },
        }
        atomic_write_json(path, payload)
    cleanup_cuda(model)


def stage_evaluate(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return
    structured_keys = parse_structured_keys(args.structured_keys)
    model = Model(args.victim_model, "4bit" if args.quantize_4bit else None)
    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "evaluation.json"
        if path.exists() and not args.overwrite:
            print(f"skip evaluation {item['sample_index']:03d} {item['id']}", flush=True)
            continue
        perturb = load_json(sample_dir(out_dir, item["sample_index"]) / "perturbation.json")
        transcript = load_json(sample_dir(out_dir, item["sample_index"]) / "transcript.json")
        candidates = load_json(sample_dir(out_dir, item["sample_index"]) / "candidates.json")
        raw_prompt = perturb["original_prompt"]
        perturbed_prompt = perturb["perturbed_prompt"]
        selection_queries = [x["user_input"] for x in transcript["selection"]]
        selection_refs = [x["protected_output"] for x in transcript["selection"]]
        eval_queries = transcript["eval_inputs"]
        print(f"evaluate sample={item['sample_index']:03d} {item['id']}", flush=True)

        cand_rows = []
        for cand in candidates["candidates"]:
            prompt_text, parsed, parsed_fields = reconstruct_prompt(cand["recovered_prompt"], structured_keys)
            if selection_queries:
                selection_outputs = generate_plain_outputs(model, prompt_text, selection_queries, gen_args(args))
                selection_score = mean(token_f1(o, r) for o, r in zip(selection_outputs, selection_refs))
            else:
                selection_outputs = []
                selection_score = 0.0
            cand_rows.append(
                {
                    "adapter": cand["adapter"],
                    "raw_recovered_prompt": cand["recovered_prompt"],
                    "executable_recovered_prompt": prompt_text,
                    "json_parse_success": parsed,
                    "parsed_fields": parsed_fields,
                    "selection_outputs": selection_outputs,
                    "selection_f1_vs_protected": selection_score,
                    "text_f1_vs_original_prompt": token_f1(prompt_text, raw_prompt),
                }
            )
        best = max(cand_rows, key=lambda x: x["selection_f1_vs_protected"])
        original_outputs = generate_plain_outputs(model, raw_prompt, eval_queries, gen_args(args))
        protected_outputs, protected_remote = generate_inferdpt_outputs(model, raw_prompt, perturbed_prompt, eval_queries, args)
        recovered_outputs = generate_plain_outputs(model, best["executable_recovered_prompt"], eval_queries, gen_args(args))
        null_outputs = generate_plain_outputs(model, "", eval_queries, gen_args(args))
        other = manifest["sample"][(item["sample_index"] + 1) % len(manifest["sample"])]
        mismatch_prompt = (result_path(other) / "original_prompt.txt").read_text(encoding="utf-8").strip()
        mismatch_outputs = generate_plain_outputs(model, mismatch_prompt, eval_queries, gen_args(args))

        per_query = []
        for idx, query in enumerate(eval_queries):
            per_query.append(
                {
                    "query_index": idx,
                    "user_input": query,
                    "original_output": original_outputs[idx],
                    "protected_output": protected_outputs[idx],
                    "protected_remote_output_not_used_by_attacker": protected_remote[idx],
                    "recovered_output": recovered_outputs[idx],
                    "null_output": null_outputs[idx],
                    "mismatch_output": mismatch_outputs[idx],
                    "f1_recovered_vs_original": token_f1(recovered_outputs[idx], original_outputs[idx]),
                    "f1_recovered_vs_protected": token_f1(recovered_outputs[idx], protected_outputs[idx]),
                    "f1_protected_vs_original": token_f1(protected_outputs[idx], original_outputs[idx]),
                    "f1_null_vs_original": token_f1(null_outputs[idx], original_outputs[idx]),
                    "f1_mismatch_vs_original": token_f1(mismatch_outputs[idx], original_outputs[idx]),
                }
            )
        row = {
            "sample_index": item["sample_index"],
            "id": item["id"],
            "category_for_manifest_only": item.get("category_for_manifest_only", ""),
            "result_dir": item["result_dir"],
            "original_prompt": raw_prompt,
            "perturbed_prompt": perturbed_prompt,
            "selected_adapter": best["adapter"],
            "selected_recovered_prompt": best["executable_recovered_prompt"],
            "selected_raw_recovered_prompt": best["raw_recovered_prompt"],
            "selected_json_parse_success": best["json_parse_success"],
            "selected_text_f1_vs_original_prompt": best["text_f1_vs_original_prompt"],
            "candidate_rerank": cand_rows,
            "eval": per_query,
            "mean_f1_recovered_vs_original": mean(x["f1_recovered_vs_original"] for x in per_query),
            "mean_f1_recovered_vs_protected": mean(x["f1_recovered_vs_protected"] for x in per_query),
            "mean_f1_protected_vs_original": mean(x["f1_protected_vs_original"] for x in per_query),
            "mean_f1_null_vs_original": mean(x["f1_null_vs_original"] for x in per_query),
            "mean_f1_mismatch_vs_original": mean(x["f1_mismatch_vs_original"] for x in per_query),
        }
        atomic_write_json(path, row)
    cleanup_cuda(model)


def summarize(rows: list[dict]) -> dict:
    summary = {
        "n_prompts": len(rows),
        "mean_text_f1_recovered_prompt_vs_original": mean(r["selected_text_f1_vs_original_prompt"] for r in rows),
        "mean_functional_f1_recovered_vs_original": mean(r["mean_f1_recovered_vs_original"] for r in rows),
        "mean_functional_f1_recovered_vs_protected": mean(r["mean_f1_recovered_vs_protected"] for r in rows),
        "mean_functional_f1_protected_vs_original": mean(r["mean_f1_protected_vs_original"] for r in rows),
        "mean_null_f1_vs_original": mean(r["mean_f1_null_vs_original"] for r in rows),
        "mean_mismatch_f1_vs_original": mean(r["mean_f1_mismatch_vs_original"] for r in rows),
        "json_parse_rate": mean(1.0 if r["selected_json_parse_success"] else 0.0 for r in rows),
        "selected_adapter_counts": dict(Counter(r["selected_adapter"] for r in rows)),
        "by_category": {},
    }
    denom = max(1e-9, summary["mean_functional_f1_protected_vs_original"] - summary["mean_mismatch_f1_vs_original"])
    summary["normalized_recovery_gain_vs_mismatch"] = (
        summary["mean_functional_f1_recovered_vs_original"] - summary["mean_mismatch_f1_vs_original"]
    ) / denom
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category_for_manifest_only"]].append(row)
    for cat, cat_rows in by_cat.items():
        summary["by_category"][cat] = {
            "n": len(cat_rows),
            "mean_functional_f1_recovered_vs_original": mean(r["mean_f1_recovered_vs_original"] for r in cat_rows),
            "mean_functional_f1_recovered_vs_protected": mean(r["mean_f1_recovered_vs_protected"] for r in cat_rows),
            "mean_text_f1_recovered_prompt_vs_original": mean(r["selected_text_f1_vs_original_prompt"] for r in cat_rows),
        }
    return summary


def stage_report(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    perturbations = collect_stage_records(out_dir, manifest, "perturbation.json")
    transcripts = collect_stage_records(out_dir, manifest, "transcript.json")
    candidates = collect_stage_records(out_dir, manifest, "candidates.json")
    evaluations = collect_stage_records(out_dir, manifest, "evaluation.json")
    summary = summarize(evaluations)
    atomic_write_json(out_dir / "perturbed_prompts.json", {"records": perturbations})
    atomic_write_json(out_dir / "evidence_transcripts.json", {"records": transcripts, "audit": audit_recovery_inputs(transcripts)})
    atomic_write_json(out_dir / "adapter_candidates.json", {"structured_keys": parse_structured_keys(args.structured_keys), "records": candidates})
    atomic_write_json(out_dir / "evaluation.json", {"records": evaluations, "summary": summary})
    atomic_write_json(out_dir / "summary.json", summary)
    report = [
        "# InferDPT Llama-3.1 Strict Black-Box Report",
        "",
        "- Defense: InferDPT open-source adaptation for hidden system prompts.",
        f"- Official repo commit: {manifest['official_repo']['commit']}",
        f"- epsilon: {manifest['epsilon']}",
        f"- Prompts: {summary['n_prompts']} / {manifest['n_prompts']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Text F1: recovered prompt vs original prompt | {summary['mean_text_f1_recovered_prompt_vs_original']:.3f} |",
        f"| Functional F1: recovered output vs original output | {summary['mean_functional_f1_recovered_vs_original']:.3f} |",
        f"| Functional F1: recovered output vs protected output | {summary['mean_functional_f1_recovered_vs_protected']:.3f} |",
        f"| Defense utility: protected output vs original output | {summary['mean_functional_f1_protected_vs_original']:.3f} |",
        f"| Null baseline: null output vs original output | {summary['mean_null_f1_vs_original']:.3f} |",
        f"| Mismatch baseline: wrong prompt output vs original output | {summary['mean_mismatch_f1_vs_original']:.3f} |",
        f"| Normalized recovery gain vs mismatch | {summary['normalized_recovery_gain_vs_mismatch']:.3f} |",
        f"| JSON parse rate | {summary['json_parse_rate']:.3f} |",
    ]
    atomic_write_text(out_dir / "REPORT.md", "\n".join(report) + "\n")
    print(out_dir / "REPORT.md", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def stage_status(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    names = ["perturbation.json", "transcript.json", "candidates.json", "evaluation.json"]
    counts = {name: sum(1 for item in manifest["sample"] if (sample_dir(out_dir, item["sample_index"]) / name).exists()) for name in names}
    print(json.dumps({"expected": manifest["n_prompts"], "completed": counts}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("results/inferdpt_llama31_e2e_20"))
    parser.add_argument("--source_manifest", type=Path, default=Path("results/strict_blackbox_step2_20/manifest.json"))
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--victim_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--recovery_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--result_prefix", default="sixcat200")
    parser.add_argument("--recovery_model_tag", default="qwen3_32b")
    parser.add_argument("--recovery_gpu", type=int, default=0)
    parser.add_argument("--structured_keys", default=",".join(DEFAULT_STRUCTURED_KEYS))
    parser.add_argument("--quantize_4bit", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--recovery_max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=6.0)
    parser.add_argument("--adjacency_size", type=int, default=64)
    parser.add_argument("--max_prompt_tokens", type=int, default=180)
    parser.add_argument("--max_subtokens_per_word", type=int, default=12)
    parser.add_argument("--keep_oov_probability", type=float, default=1.0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--stage", choices=["manifest", "perturb", "collect", "recover", "evaluate", "worker", "report", "status"], default="worker")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_manifest", action="store_true")
    parser.add_argument("--save_recovery_inputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    random.seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)
    out_dir = resolve_output(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_manifest(args)
    start = time.time()
    if args.stage == "manifest":
        print(out_dir / "manifest.json", flush=True)
    elif args.stage == "perturb":
        stage_perturb(args, manifest)
    elif args.stage == "collect":
        stage_collect(args, manifest)
    elif args.stage == "recover":
        stage_recover(args, manifest)
    elif args.stage == "evaluate":
        stage_evaluate(args, manifest)
    elif args.stage == "worker":
        stage_perturb(args, manifest)
        stage_collect(args, manifest)
        stage_recover(args, manifest)
        stage_evaluate(args, manifest)
    elif args.stage == "report":
        stage_report(args, manifest)
    elif args.stage == "status":
        stage_status(args, manifest)
    print(f"stage={args.stage} shard={args.shard_index}/{args.num_shards} done in {time.time() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
