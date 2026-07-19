#!/usr/bin/env python3
"""End-to-end EmojiPrompt adaptation for strict black-box prompt recovery.

The official EmojiPrompt repository is notebook-only. This runner implements
the paper's core mechanism for system-prompt experiments:

1. split a system prompt into atomic text units;
2. use an obfuscation LLM to convert each unit into emoji/symbol text;
3. enforce the paper's adjacency/semantic-alignment rule where possible;
4. optionally apply epsilon-LDP post-sampling on adjacent cliques;
5. query a victim model that uses the obfuscated prompt as its system prompt;
6. run the frozen six-adapter Qwen recovery model from the Step 2 pipeline;
7. evaluate text F1 and functional utility F1.

This is an open-source adaptation when --obfuscator_model/--victim_model are
local models such as Llama-3.1 instead of the paper's GPT-4/Gemini setup.
"""

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

import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "recovery_tools"))

from recovery_tools.run_strict_blackbox_step2 import (  # noqa: E402
    CATEGORIES,
    DEFAULT_STRUCTURED_KEYS,
    audit_recovery_inputs,
    build_blackbox_recovery_prompt,
    generate_plain_outputs,
    generate_recovery_text,
    load_json,
    load_recovery_base,
    load_test_queries,
    mean,
    parse_structured_keys,
    reconstruct_prompt,
    result_path,
    token_f1,
    write_json,
)
from src.model import Model  # noqa: E402
from src.utils import set_seed  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+|(?<=,)\s+(?=[A-Z0-9])")


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sample_dir(output_dir: Path, sample_index: int) -> Path:
    return output_dir / "samples" / f"{sample_index:03d}"


def cleanup_cuda(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_atomic_units(text: str, max_units: int) -> list[str]:
    chunks = [x.strip() for x in SPLIT_RE.split(text) if x.strip()]
    if not chunks:
        chunks = [text.strip()]
    if len(chunks) <= max_units:
        return chunks

    merged = chunks[: max_units - 1]
    merged.append(" ".join(chunks[max_units - 1 :]))
    return merged


def token_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i]
        for j, tb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


def adjacent(x1: str, x2: str, rho: float) -> bool:
    t1 = WORD_RE.findall(x1.lower())
    t2 = WORD_RE.findall(x2.lower())
    if not t1 or not t2:
        return False
    threshold = math.ceil(rho * max(len(t1), len(t2)))
    return edit_distance(t1, t2) <= threshold


class Similarity:
    def __init__(self, metric: str, device: str):
        self.metric = metric
        self.scorer = None
        if metric == "bertscore":
            from bert_score import BERTScorer

            self.scorer = BERTScorer(
                model_type="roberta-large",
                lang="en",
                rescale_with_baseline=False,
                device=device,
            )

    def score(self, left: str, right: str) -> float:
        if not left.strip() or not right.strip():
            return 0.0
        if self.metric == "token_f1":
            return token_f1(left, right)
        if self.scorer is None:
            return 1.0
        with torch.no_grad():
            _, _, f1 = self.scorer.score([left], [right])
        return float(f1[0].item())


def llm_generation_args(args: argparse.Namespace, max_new_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        victim_batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def call_llm(model: Model, system_prompt: str, user_prompt: str, args: argparse.Namespace, max_new_tokens: int) -> str:
    gen_args = llm_generation_args(args, max_new_tokens=max_new_tokens)
    return generate_plain_outputs(model, system_prompt, [user_prompt], gen_args)[0].strip()


def extract_obfuscation(text: str) -> str:
    cleaned = text.strip()
    patterns = [
        r"(?is)obfuscated prompt\s*:\s*(.*)",
        r"(?is)obfuscation\s*:\s*(.*)",
        r"(?is)protected prompt\s*:\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            cleaned = match.group(1).strip()
            break
    cleaned = re.split(r"(?is)\n\s*explanation\s*:", cleaned)[0].strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", cleaned).strip()
    return cleaned or text.strip()


def obfuscator_system_prompt() -> str:
    return (
        "You are an obfuscation LLM implementing EmojiPrompt. Transform input text "
        "into a compact non-natural-language sequence made of emojis, symbols, "
        "numbers, arrows, operators, abbreviations, and short fragments. Preserve "
        "the operational meaning for another instruction-following LLM. Do not "
        "explain. Output only the obfuscated sequence."
    )


def obfuscator_user_prompt(unit: str) -> str:
    return (
        "Transform this system-prompt unit into an EmojiPrompt-style protected "
        "sequence. Keep task intent, constraints, formatting rules, and tool "
        "protocol semantics recoverable for an LLM, but avoid ordinary prose.\n\n"
        f"Unit:\n{unit}\n\nObfuscated sequence:"
    )


def semantic_ok(
    candidate: str,
    unit: str,
    prior_units: list[str],
    prior_obfuscated: list[str],
    sim: Similarity,
    epsilon: float,
    rho: float,
) -> bool:
    for prev_unit, prev_obf in zip(prior_units, prior_obfuscated):
        if not adjacent(unit, prev_unit, rho):
            continue
        original_sim = sim.score(unit, prev_unit)
        obfuscated_sim = sim.score(candidate, prev_obf)
        if obfuscated_sim + 1e-6 < original_sim / epsilon:
            return False
    return True


def obfuscate_unit(
    model: Model,
    unit: str,
    prior_units: list[str],
    prior_obfuscated: list[str],
    sim: Similarity,
    args: argparse.Namespace,
) -> tuple[str, list[dict]]:
    attempts = []
    best_text = ""
    best_score = -1.0
    system = obfuscator_system_prompt()
    prompt = obfuscator_user_prompt(unit)
    for attempt in range(1, args.max_obfuscation_attempts + 1):
        raw = call_llm(model, system, prompt, args, args.obfuscation_max_new_tokens)
        candidate = extract_obfuscation(raw)
        scores = []
        for prev_unit, prev_obf in zip(prior_units, prior_obfuscated):
            if adjacent(unit, prev_unit, args.rho):
                scores.append(sim.score(candidate, prev_obf))
        mean_score = mean(scores) if scores else 1.0
        ok = semantic_ok(candidate, unit, prior_units, prior_obfuscated, sim, args.semantic_epsilon, args.rho)
        attempts.append(
            {
                "attempt": attempt,
                "raw": raw,
                "candidate": candidate,
                "mean_adjacent_obfuscated_similarity": mean_score,
                "accepted": ok,
            }
        )
        if mean_score > best_score:
            best_text = candidate
            best_score = mean_score
        if ok:
            return candidate, attempts
    return best_text, attempts


def adjacency_matrix(units: list[str], rho: float) -> list[list[bool]]:
    n = len(units)
    matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[j][i] = adjacent(units[i], units[j], rho)
    return matrix


def greedy_max_clique(nodes: set[int], matrix: list[list[bool]]) -> list[int]:
    clique: list[int] = []
    candidates = set(nodes)
    while candidates:
        node = max(candidates, key=lambda n: sum(1 for m in candidates if matrix[n][m]))
        clique.append(node)
        candidates = {m for m in candidates if m != node and all(matrix[m][c] for c in clique)}
    return clique


def post_sample_obfuscations(obfuscated: list[str], units: list[str], args: argparse.Namespace) -> tuple[list[str], list[dict]]:
    if not args.post_sampling or len(obfuscated) <= 1:
        return obfuscated, []

    rng = random.Random(args.seed)
    matrix = adjacency_matrix(units, args.rho)
    remaining = set(range(len(units)))
    final = list(obfuscated)
    events = []

    while remaining:
        clique = greedy_max_clique(remaining, matrix)
        if len(clique) <= 1:
            break
        weight_self = math.exp(args.post_sampling_epsilon)
        denominator = weight_self + len(clique) - 1
        for node in clique:
            draw = rng.random()
            cumulative = 0.0
            chosen = node
            for candidate_node in clique:
                weight = weight_self if candidate_node == node else 1.0
                cumulative += weight / denominator
                if draw <= cumulative:
                    chosen = candidate_node
                    break
            final[node] = obfuscated[chosen]
            events.append(
                {
                    "node": node,
                    "clique": clique,
                    "chosen_obfuscation_from_node": chosen,
                    "self_probability": weight_self / denominator,
                }
            )
        remaining -= set(clique)

    return final, events


def load_source_manifest(args: argparse.Namespace) -> dict:
    source = args.source_manifest if args.source_manifest.is_absolute() else ROOT / args.source_manifest
    return load_json(source)


def stage_manifest(args: argparse.Namespace) -> dict:
    out_dir = resolve_output(args.output_dir)
    out_path = out_dir / "manifest.json"
    if out_path.exists() and not args.overwrite_manifest:
        return load_json(out_path)

    source = load_source_manifest(args)
    manifest = {
        "setting": "emojiprompt_llama31_strict_blackbox",
        "description": (
            "EmojiPrompt open-source adaptation for hidden system prompts. "
            "Recovery inputs contain only user inputs and protected model outputs."
        ),
        "source_manifest": str(args.source_manifest),
        "official_repo": {
            "path": "third_party/EmojiCrypt",
            "commit": read_git_commit(ROOT / "third_party" / "EmojiCrypt"),
        },
        "obfuscator_model": args.obfuscator_model,
        "victim_model": args.victim_model,
        "recovery_model": args.recovery_model,
        "recovery_result_prefix": args.result_prefix,
        "recovery_model_tag": args.recovery_model_tag,
        "n_prompts": min(args.n_prompts, len(source["sample"])),
        "evidence_queries": source["evidence_queries"],
        "selection_queries": source["selection_queries"],
        "eval_queries": source["eval_queries"],
        "rho": args.rho,
        "semantic_epsilon": args.semantic_epsilon,
        "post_sampling": args.post_sampling,
        "post_sampling_epsilon": args.post_sampling_epsilon,
        "semantic_metric": args.semantic_metric,
        "sample": source["sample"][: args.n_prompts],
    }
    atomic_write_json(out_path, manifest)
    return manifest


def read_git_commit(path: Path) -> str:
    head = path / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = content.split(" ", 1)[1]
        ref_path = path / ".git" / ref
        return ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else "unknown"
    return content


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def shard_items(items: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.num_shards <= 1:
        return items
    return [x for x in items if x["sample_index"] % args.num_shards == args.shard_index]


def stage_obfuscate(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return

    set_seed(args.seed + args.shard_index)
    model = Model(args.obfuscator_model, "4bit" if args.quantize_4bit else None)
    sim_device = "cuda" if torch.cuda.is_available() and args.semantic_metric == "bertscore" else "cpu"
    sim = Similarity(args.semantic_metric, sim_device)

    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "obfuscation.json"
        if path.exists() and not args.overwrite:
            print(f"skip obfuscation {item['sample_index']:03d} {item['id']}", flush=True)
            continue

        original_prompt = (result_path(item) / "original_prompt.txt").read_text(encoding="utf-8").strip()
        units = split_atomic_units(original_prompt, args.max_atomic_units)
        order = list(range(len(units)))
        random.Random(args.seed + item["sample_index"]).shuffle(order)
        ordered_units = [units[i] for i in order]

        ordered_obfuscated: list[str] = []
        unit_records = []
        for ordered_index, unit in enumerate(ordered_units):
            print(
                f"obfuscate sample={item['sample_index']:03d} unit={ordered_index + 1}/{len(ordered_units)}",
                flush=True,
            )
            obfuscated, attempts = obfuscate_unit(model, unit, ordered_units[:ordered_index], ordered_obfuscated, sim, args)
            ordered_obfuscated.append(obfuscated)
            unit_records.append(
                {
                    "ordered_index": ordered_index,
                    "original_unit_index": order[ordered_index],
                    "unit": unit,
                    "obfuscated": obfuscated,
                    "attempts": attempts,
                }
            )

        final_ordered, post_sampling_events = post_sample_obfuscations(ordered_obfuscated, ordered_units, args)
        final_by_original = [""] * len(units)
        for ordered_index, original_index in enumerate(order):
            final_by_original[original_index] = final_ordered[ordered_index]

        obfuscated_prompt = "\n".join(final_by_original).strip()
        payload = {
            "sample_index": item["sample_index"],
            "id": item["id"],
            "original_prompt": original_prompt,
            "atomic_units": units,
            "shuffle_order": order,
            "unit_records": unit_records,
            "post_sampling_events": post_sampling_events,
            "obfuscated_prompt": obfuscated_prompt,
            "text_f1_obfuscated_vs_original": token_f1(obfuscated_prompt, original_prompt),
        }
        atomic_write_json(path, payload)
        atomic_write_text(sample_dir(out_dir, item["sample_index"]) / "obfuscated_prompt.txt", obfuscated_prompt)

    cleanup_cuda(model, sim)


def stage_collect(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return

    set_seed(args.seed + args.shard_index)
    model = Model(args.victim_model, "4bit" if args.quantize_4bit else None)
    gen_args = llm_generation_args(args, args.max_new_tokens)

    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "transcript.json"
        if path.exists() and not args.overwrite:
            print(f"skip transcript {item['sample_index']:03d} {item['id']}", flush=True)
            continue

        obf_path = sample_dir(out_dir, item["sample_index"]) / "obfuscation.json"
        if not obf_path.exists():
            raise FileNotFoundError(f"Missing obfuscation for sample {item['sample_index']}: {obf_path}")
        obfuscated_prompt = load_json(obf_path)["obfuscated_prompt"]
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
        evidence_outputs = generate_plain_outputs(model, obfuscated_prompt, evidence, gen_args)
        selection_outputs = generate_plain_outputs(model, obfuscated_prompt, selection, gen_args) if selection else []
        payload = {
            "sample_index": item["sample_index"],
            "id": item["id"],
            "evidence": [
                {"user_input": q, "protected_output": o}
                for q, o in zip(evidence, evidence_outputs)
            ],
            "selection": [
                {"user_input": q, "protected_output": o}
                for q, o in zip(selection, selection_outputs)
            ],
            "eval_inputs": eval_queries,
        }
        atomic_write_json(path, payload)

    cleanup_cuda(model)


def load_recovery_model(args: argparse.Namespace):
    tokenizer, base = load_recovery_base(args)
    adapter_paths = {
        category: ROOT / "results" / f"{args.result_prefix}_{category}_structured_{args.recovery_model_tag}_lora" / "adapter"
        for category in CATEGORIES
    }
    first = CATEGORIES[0]
    model = PeftModel.from_pretrained(base, adapter_paths[first], adapter_name=first)
    for category in CATEGORIES[1:]:
        model.load_adapter(adapter_paths[category], adapter_name=category)
    model.eval()
    return tokenizer, model


def stage_recover(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return

    structured_keys = parse_structured_keys(args.structured_keys)
    tokenizer, model = load_recovery_model(args)
    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "candidates.json"
        if path.exists() and not args.overwrite:
            print(f"skip recovery {item['sample_index']:03d} {item['id']}", flush=True)
            continue

        transcript_path = sample_dir(out_dir, item["sample_index"]) / "transcript.json"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript for sample {item['sample_index']}: {transcript_path}")
        transcript = load_json(transcript_path)
        recovery_input = build_blackbox_recovery_prompt(transcript, structured_keys)
        record = {
            "sample_index": item["sample_index"],
            "id": item["id"],
            "recovery_input": recovery_input if args.save_recovery_inputs else None,
            "candidates": [],
        }
        for category in CATEGORIES:
            print(f"recover sample={item['sample_index']:03d} adapter={category}", flush=True)
            model.set_adapter(category)
            recovered = generate_recovery_text(tokenizer, model, recovery_input, args, structured_keys)
            record["candidates"].append({"adapter": category, "recovered_prompt": recovered})
        atomic_write_json(path, record)

    cleanup_cuda(model)


def stage_evaluate(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    items = shard_items(manifest["sample"], args)
    if not items:
        return

    set_seed(args.seed + args.shard_index)
    structured_keys = parse_structured_keys(args.structured_keys)
    model = Model(args.victim_model, "4bit" if args.quantize_4bit else None)
    gen_args = llm_generation_args(args, args.max_new_tokens)

    for item in items:
        path = sample_dir(out_dir, item["sample_index"]) / "evaluation.json"
        if path.exists() and not args.overwrite:
            print(f"skip evaluation {item['sample_index']:03d} {item['id']}", flush=True)
            continue

        transcript = load_json(sample_dir(out_dir, item["sample_index"]) / "transcript.json")
        candidates = load_json(sample_dir(out_dir, item["sample_index"]) / "candidates.json")
        obf = load_json(sample_dir(out_dir, item["sample_index"]) / "obfuscation.json")
        original_prompt = obf["original_prompt"]
        obfuscated_prompt = obf["obfuscated_prompt"]
        selection_queries = [x["user_input"] for x in transcript["selection"]]
        selection_refs = [x["protected_output"] for x in transcript["selection"]]
        eval_queries = transcript["eval_inputs"]

        print(f"evaluate sample={item['sample_index']:03d} {item['id']}", flush=True)
        cand_rows = []
        for cand in candidates["candidates"]:
            prompt_text, parsed, parsed_fields = reconstruct_prompt(cand["recovered_prompt"], structured_keys)
            if selection_queries:
                selection_outputs = generate_plain_outputs(model, prompt_text, selection_queries, gen_args)
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
                    "text_f1_vs_original_prompt": token_f1(prompt_text, original_prompt),
                }
            )

        best = max(cand_rows, key=lambda x: x["selection_f1_vs_protected"])
        original_outputs = generate_plain_outputs(model, original_prompt, eval_queries, gen_args)
        protected_outputs = generate_plain_outputs(model, obfuscated_prompt, eval_queries, gen_args)
        recovered_outputs = generate_plain_outputs(model, best["executable_recovered_prompt"], eval_queries, gen_args)
        null_outputs = generate_plain_outputs(model, "", eval_queries, gen_args)

        other = manifest["sample"][(item["sample_index"] + 1) % len(manifest["sample"])]
        mismatch_prompt = (result_path(other) / "original_prompt.txt").read_text(encoding="utf-8").strip()
        mismatch_outputs = generate_plain_outputs(model, mismatch_prompt, eval_queries, gen_args)

        per_query = []
        for idx, query in enumerate(eval_queries):
            per_query.append(
                {
                    "query_index": idx,
                    "user_input": query,
                    "original_output": original_outputs[idx],
                    "protected_output": protected_outputs[idx],
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
            "original_prompt": original_prompt,
            "obfuscated_prompt": obfuscated_prompt,
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


def collect_stage_records(output_dir: Path, manifest: dict, filename: str) -> list[dict]:
    rows = []
    for item in manifest["sample"]:
        path = sample_dir(output_dir, item["sample_index"]) / filename
        if path.exists():
            rows.append(load_json(path))
    return sorted(rows, key=lambda x: x["sample_index"])


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


def write_report(out_dir: Path, manifest: dict, evaluations: list[dict], audit: dict) -> None:
    summary = summarize(evaluations)
    lines = [
        "# EmojiPrompt Llama-3.1 Strict Black-Box Report",
        "",
        "## Setting",
        "",
        "- Defense: EmojiPrompt-style open-source adaptation for hidden system prompts.",
        f"- Obfuscator model: {manifest['obfuscator_model']}",
        f"- Victim model: {manifest['victim_model']}",
        f"- Recovery model: {manifest['recovery_model']} with six frozen LoRA adapters from {manifest['recovery_result_prefix']}.",
        "- Recovery input: user input + protected model output only.",
        "- Original prompts, labels, categories, and obfuscated prompts are used only for evaluation/reporting.",
        f"- Prompts evaluated: {summary['n_prompts']} / {manifest['n_prompts']}",
        f"- Evidence / selection / eval queries: {manifest['evidence_queries']} / {manifest['selection_queries']} / {manifest['eval_queries']}",
        f"- Semantic metric: {manifest['semantic_metric']}; rho={manifest['rho']}; semantic epsilon={manifest['semantic_epsilon']}",
        f"- Post-sampling: {manifest['post_sampling']} with epsilon={manifest['post_sampling_epsilon']}",
        "",
        "## Isolation Audit",
        "",
        f"- Forbidden term hit in recovery prompts: {audit['contains_forbidden_terms']}",
        f"- Evidence-selection overlap: {audit['evidence_selection_overlap']}",
        f"- Evidence-eval overlap: {audit['evidence_eval_overlap']}",
        f"- Selection-eval overlap: {audit['selection_eval_overlap']}",
        "",
        "## Main Results",
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
        "",
        "## By Category",
        "",
        "| Category | N | Functional F1 vs original | Functional F1 vs protected | Text F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat in CATEGORIES:
        if cat not in summary["by_category"]:
            continue
        row = summary["by_category"][cat]
        lines.append(
            f"| {cat} | {row['n']} | {row['mean_functional_f1_recovered_vs_original']:.3f} | "
            f"{row['mean_functional_f1_recovered_vs_protected']:.3f} | {row['mean_text_f1_recovered_prompt_vs_original']:.3f} |"
        )
    atomic_write_text(out_dir / "REPORT.md", "\n".join(lines) + "\n")
    atomic_write_json(out_dir / "summary.json", summary)


def stage_report(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    obfuscations = collect_stage_records(out_dir, manifest, "obfuscation.json")
    transcripts = collect_stage_records(out_dir, manifest, "transcript.json")
    candidates = collect_stage_records(out_dir, manifest, "candidates.json")
    evaluations = collect_stage_records(out_dir, manifest, "evaluation.json")
    atomic_write_json(out_dir / "obfuscated_prompts.json", {"records": obfuscations})
    atomic_write_json(
        out_dir / "evidence_transcripts.json",
        {"records": transcripts, "audit": audit_recovery_inputs(transcripts)},
    )
    atomic_write_json(out_dir / "adapter_candidates.json", {"structured_keys": parse_structured_keys(args.structured_keys), "records": candidates})
    atomic_write_json(out_dir / "evaluation.json", {"records": evaluations, "summary": summarize(evaluations)})
    write_report(out_dir, manifest, evaluations, audit_recovery_inputs(transcripts))
    print(out_dir / "REPORT.md", flush=True)
    print(json.dumps(summarize(evaluations), indent=2, ensure_ascii=False), flush=True)


def stage_status(args: argparse.Namespace, manifest: dict) -> None:
    out_dir = resolve_output(args.output_dir)
    names = ["obfuscation.json", "transcript.json", "candidates.json", "evaluation.json"]
    counts = {}
    for name in names:
        counts[name] = sum(1 for item in manifest["sample"] if (sample_dir(out_dir, item["sample_index"]) / name).exists())
    print(json.dumps({"expected": manifest["n_prompts"], "completed": counts}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("results/emojiprompt_llama31_e2e_20"))
    parser.add_argument("--source_manifest", type=Path, default=Path("results/strict_blackbox_step2_20/manifest.json"))
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obfuscator_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--victim_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--recovery_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--result_prefix", default="sixcat200")
    parser.add_argument("--recovery_model_tag", default="qwen3_32b")
    parser.add_argument("--recovery_gpu", type=int, default=0)
    parser.add_argument("--structured_keys", default=",".join(DEFAULT_STRUCTURED_KEYS))
    parser.add_argument("--quantize_4bit", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--obfuscation_max_new_tokens", type=int, default=220)
    parser.add_argument("--recovery_max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_atomic_units", type=int, default=24)
    parser.add_argument("--max_obfuscation_attempts", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--semantic_epsilon", type=float, default=10.0)
    parser.add_argument("--semantic_metric", choices=["bertscore", "token_f1", "none"], default="bertscore")
    parser.add_argument("--post_sampling", action="store_true", default=True)
    parser.add_argument("--no_post_sampling", action="store_false", dest="post_sampling")
    parser.add_argument("--post_sampling_epsilon", type=float, default=10.0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--stage",
        choices=["manifest", "obfuscate", "collect", "recover", "evaluate", "worker", "report", "status"],
        default="worker",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_manifest", action="store_true")
    parser.add_argument("--save_recovery_inputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.semantic_metric == "none":
        args.semantic_metric = "token_f1"
        args.semantic_epsilon = 1e9

    random.seed(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)
    out_dir = resolve_output(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_manifest(args)

    start = time.time()
    if args.stage == "manifest":
        print(out_dir / "manifest.json", flush=True)
    elif args.stage == "obfuscate":
        stage_obfuscate(args, manifest)
    elif args.stage == "collect":
        stage_collect(args, manifest)
    elif args.stage == "recover":
        stage_recover(args, manifest)
    elif args.stage == "evaluate":
        stage_evaluate(args, manifest)
    elif args.stage == "worker":
        stage_obfuscate(args, manifest)
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
