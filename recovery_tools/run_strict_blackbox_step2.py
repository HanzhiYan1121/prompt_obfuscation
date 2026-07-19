#!/usr/bin/env python3
"""Strict black-box Step 2 prompt-recovery runner.

This script evaluates the end-to-end attack setting where the recovery model
only sees user inputs and outputs from the protected victim model. It does not
put prompt labels, categories, original prompts, soft prompts, projections, or
metadata in the recovery-model input.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "recovery_tools"))

from data.utils import TextDataset, create_collate_fn
from recovery_tools.eval_acl100_recovery_lora import parse_json_object
from recovery_tools.train_acl100_recovery_lora import apply_chat_template, parse_structured_keys, structured_system_message
from src.model import Model
from src.output_generation import generate_model_responses, generate_model_responses_replace
from src.utils import set_seed


CATEGORIES = [
    "role_persona",
    "task_instruction",
    "format_spec",
    "constraints",
    "context_knowledge",
    "tool_protocol",
]

DEFAULT_STRUCTURED_KEYS = ["role", "task", "format", "constraints", "context", "tool_protocol"]
FORBIDDEN_RECOVERY_INPUT_TERMS = [
    "prompt_label",
    "Prompt label:",
    "paper_title",
    "Paper title:",
    "Six-dimension scores:",
    "Nearest-token projections",
    "Euclidean:",
    "Cosine:",
    "best_candidate",
    "result_dir",
    "target_original_prompt",
]

WORD_RE = re.compile(r"[a-z0-9]+")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def flatten_output(group) -> str:
    if isinstance(group, list):
        return "\n".join(str(x) for x in group)
    return str(group)


def tokens(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def token_f1(pred: str, ref: str) -> float:
    pc = Counter(tokens(pred))
    rc = Counter(tokens(ref))
    if not pc or not rc:
        return 0.0
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pc.values())
    recall = overlap / sum(rc.values())
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_quotas(text: str, n: int) -> dict[str, int]:
    if text:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
        if len(values) != len(CATEGORIES):
            raise ValueError("--category_quotas must contain six comma-separated integers")
        quotas = dict(zip(CATEGORIES, values))
    else:
        base = n // len(CATEGORIES)
        extra = n % len(CATEGORIES)
        quotas = {cat: base + (1 if i < extra else 0) for i, cat in enumerate(CATEGORIES)}
    if sum(quotas.values()) != n:
        raise ValueError(f"Category quotas sum to {sum(quotas.values())}, not --n_prompts={n}")
    return quotas


def generation_args(args: argparse.Namespace) -> dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": 1,
    }


def result_path(record: dict) -> Path:
    return resolve(Path(record["result_dir"]))


def load_test_queries(result_dir: Path, needed: int) -> list[str]:
    path = result_dir / "prepared_data" / "test_data.json"
    queries = [str(x) for x in load_json(path)]
    if len(queries) < needed:
        raise ValueError(f"{result_dir} has only {len(queries)} test queries; need {needed}")
    return queries


def stage_manifest(args: argparse.Namespace) -> dict:
    out_dir = resolve(args.output_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        return load_json(manifest_path)

    quotas = parse_quotas(args.category_quotas, args.n_prompts)
    selected = []
    seen_ids = set()
    skipped = defaultdict(int)
    needed_queries = args.evidence_queries + args.selection_queries + args.eval_queries

    train_ids_by_category = {}
    test_ids_by_category = {}
    by_category = {}
    for category in CATEGORIES:
        inputs_path = ROOT / "results" / f"{args.result_prefix}_{category}_structured_recovery" / "recovery_inputs.json"
        lora_dir = ROOT / "results" / f"{args.result_prefix}_{category}_structured_{args.recovery_model_tag}_lora"
        split_path = lora_dir / "split.json"
        records = load_json(inputs_path)
        split = load_json(split_path)
        train_ids_by_category[category] = set(split["train_ids"])
        test_ids_by_category[category] = set(split["test_ids"])
    all_train_ids = set().union(*train_ids_by_category.values())

    for category in CATEGORIES:
        inputs_path = ROOT / "results" / f"{args.result_prefix}_{category}_structured_recovery" / "recovery_inputs.json"
        records = load_json(inputs_path)
        test_ids = test_ids_by_category[category]
        rows = []
        for record in sorted(records, key=lambda r: r["id"]):
            if record["id"] not in test_ids:
                skipped["not_heldout"] += 1
                continue
            if record["id"] in all_train_ids:
                skipped["in_some_adapter_train_set"] += 1
                continue
            rdir = result_path(record)
            if not (rdir / "best_candidate.pt").exists():
                skipped["missing_best_candidate"] += 1
                continue
            if not (rdir / "original_prompt.txt").exists():
                skipped["missing_original_prompt"] += 1
                continue
            try:
                load_test_queries(rdir, needed_queries)
            except Exception:
                skipped["not_enough_queries"] += 1
                continue
            rows.append(record)
        by_category[category] = rows

    for category in CATEGORIES:
        count = 0
        for record in by_category[category]:
            if record["id"] in seen_ids:
                skipped["duplicate_id"] += 1
                continue
            selected.append(
                {
                    "sample_index": len(selected),
                    "id": record["id"],
                    "category_for_manifest_only": category,
                    "result_dir": str(Path(record["result_dir"])),
                    "adapter_train_membership": {
                        cat: record["id"] in train_ids_by_category[cat]
                        for cat in CATEGORIES
                    },
                }
            )
            seen_ids.add(record["id"])
            count += 1
            if count >= quotas[category]:
                break

    if len(selected) < args.n_prompts:
        raise RuntimeError(f"Could only select {len(selected)} prompts; requested {args.n_prompts}")

    manifest = {
        "setting": "strict_blackbox_step2",
        "description": (
            "Recovery inputs contain only attacker-observed user inputs and protected model outputs. "
            "Labels, categories, original prompts, soft prompts, projections, scores, and metadata are not supplied."
        ),
        "seed": args.seed,
        "n_prompts": args.n_prompts,
        "category_quotas": quotas,
        "result_prefix": args.result_prefix,
        "recovery_model_tag": args.recovery_model_tag,
        "victim_model": args.victim_model,
        "recovery_model": args.recovery_model,
        "evidence_queries": args.evidence_queries,
        "selection_queries": args.selection_queries,
        "eval_queries": args.eval_queries,
        "sample": selected[: args.n_prompts],
        "skipped_counts": dict(skipped),
    }
    write_json(manifest_path, manifest)
    return manifest


def generate_plain_outputs(model: Model, system_prompt: str, queries: list[str], args: argparse.Namespace) -> list[str]:
    dataset = TextDataset(queries)
    collate_fn = create_collate_fn(model.tokenizer, system_prompt)
    loader = DataLoader(dataset, batch_size=args.victim_batch_size, collate_fn=collate_fn)
    raw = generate_model_responses(model, loader, generation_args(args))
    return [flatten_output(x) for x in raw]


def generate_soft_outputs(model: Model, result_dir: Path, queries: list[str], args: argparse.Namespace) -> list[str]:
    params = load_json(result_dir / "params.json")
    system_prompt = params["system_prompt"]
    dataset = TextDataset(queries)
    collate_fn = create_collate_fn(model.tokenizer, system_prompt)
    loader = DataLoader(dataset, batch_size=args.victim_batch_size, collate_fn=collate_fn)
    system_prompt_ids = model.tokenizer(system_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    best_embedding = torch.load(result_dir / "best_candidate.pt", map_location="cpu", weights_only=True)
    raw = generate_model_responses_replace(
        model_wrapper=model,
        dataloader=loader,
        generation_args=generation_args(args),
        sys_prompt_obf=best_embedding,
        original_sys_prompt_ids=system_prompt_ids,
        is_soft_prompt_method=(params["obfuscation_method"] == "soft"),
        obf_sys_prompt_len=params["obf_sys_prompt_len"],
        pad_token_id=model.tokenizer.pad_token_id,
    )
    return [flatten_output(x) for x in raw]


def stage_collect(args: argparse.Namespace, manifest: dict) -> dict:
    out_dir = resolve(args.output_dir)
    transcript_path = out_dir / "evidence_transcripts.json"
    if transcript_path.exists() and not args.overwrite:
        return load_json(transcript_path)

    set_seed(args.seed)
    model = Model(args.victim_model, "4bit" if args.victim_quantize_4bit else None)
    records = []
    for item in manifest["sample"]:
        result_dir = result_path(item)
        all_queries = load_test_queries(
            result_dir,
            args.evidence_queries + args.selection_queries + args.eval_queries,
        )
        evidence = all_queries[: args.evidence_queries]
        selection = all_queries[args.evidence_queries : args.evidence_queries + args.selection_queries]
        eval_queries = all_queries[
            args.evidence_queries + args.selection_queries : args.evidence_queries + args.selection_queries + args.eval_queries
        ]
        print(f"Collecting protected outputs: {item['id']}", flush=True)
        evidence_outputs = generate_soft_outputs(model, result_dir, evidence, args)
        selection_outputs = generate_soft_outputs(model, result_dir, selection, args) if selection else []
        records.append(
            {
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
        )
    payload = {
        "generation_config": generation_args(args),
        "records": records,
        "audit": audit_recovery_inputs(records),
    }
    write_json(transcript_path, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def build_blackbox_recovery_prompt(record: dict, structured_keys: list[str]) -> str:
    template = "{" + ", ".join(f'"{key}": "..."' for key in structured_keys) + "}"
    lines = [
        "Recover the hidden system prompt using only the black-box behavior below.",
        "The behavior comes from a protected model whose system prompt is hidden.",
        "Use only the transcript below. Do not add outside facts.",
        "",
        "Observed behavior:",
    ]
    for index, example in enumerate(record["evidence"], start=1):
        lines.extend(
            [
                f"Example {index}",
                f"User input: {example['user_input']}",
                f"Protected model output: {example['protected_output']}",
            ]
        )
    lines.extend(
        [
            "",
            "Return a JSON object with exactly these keys:",
            template,
            "Use empty strings when a field is not clear. Output JSON only.",
        ]
    )
    return "\n".join(lines)


def audit_recovery_inputs(records: list[dict]) -> dict:
    prompts = [build_blackbox_recovery_prompt(r, DEFAULT_STRUCTURED_KEYS) for r in records]
    joined = "\n".join(prompts).lower()
    term_hits = {term: (term.lower() in joined) for term in FORBIDDEN_RECOVERY_INPUT_TERMS}
    evidence_inputs = {
        (r["id"], ex["user_input"])
        for r in records
        for ex in r["evidence"]
    }
    selection_inputs = {
        (r["id"], ex["user_input"])
        for r in records
        for ex in r["selection"]
    }
    eval_inputs = {
        (r["id"], q)
        for r in records
        for q in r["eval_inputs"]
    }
    return {
        "forbidden_term_hits_in_recovery_prompts": term_hits,
        "contains_forbidden_terms": any(term_hits.values()),
        "evidence_selection_overlap": len(evidence_inputs & selection_inputs),
        "evidence_eval_overlap": len(evidence_inputs & eval_inputs),
        "selection_eval_overlap": len(selection_inputs & eval_inputs),
        "n_recovery_prompts": len(prompts),
    }


def load_recovery_base(args: argparse.Namespace):
    cache = os.environ.get("TRANSFORMERS_CACHE", "/scratch/Hanzhi/.cache/huggingface")
    visible_devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    local_gpu = 0 if visible_devices else args.recovery_gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.recovery_model, cache_dir=cache, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.recovery_model,
        quantization_config=bnb,
        device_map={"": f"cuda:{local_gpu}"},
        cache_dir=cache,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_recovery_text(tokenizer, model, prompt: str, args: argparse.Namespace, structured_keys: list[str]) -> str:
    messages = [
        {"role": "system", "content": structured_system_message(structured_keys)},
        {"role": "user", "content": prompt},
    ]
    ids = apply_chat_template(tokenizer, messages, tokenize=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(ids, device=model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=args.recovery_max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def stage_recover(args: argparse.Namespace, transcripts: dict) -> dict:
    out_dir = resolve(args.output_dir)
    candidates_path = out_dir / "adapter_candidates.json"
    if candidates_path.exists() and not args.overwrite:
        return load_json(candidates_path)

    structured_keys = parse_structured_keys(args.structured_keys)
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

    records = []
    for record in transcripts["records"]:
        prompt = build_blackbox_recovery_prompt(record, structured_keys)
        item = {
            "sample_index": record["sample_index"],
            "id": record["id"],
            "recovery_input": prompt if args.save_recovery_inputs else None,
            "candidates": [],
        }
        for category in CATEGORIES:
            print(f"Recovering candidate with {category}: {record['id']}", flush=True)
            model.set_adapter(category)
            recovered = generate_recovery_text(tokenizer, model, prompt, args, structured_keys)
            item["candidates"].append({"adapter": category, "recovered_prompt": recovered})
        records.append(item)

    payload = {
        "structured_keys": structured_keys,
        "audit": audit_recovery_inputs(transcripts["records"]),
        "records": records,
    }
    write_json(candidates_path, payload)
    del model
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def reconstruct_prompt(recovered: str, structured_keys: list[str]) -> tuple[str, bool, dict]:
    obj = parse_json_object(recovered) or None
    if not obj:
        return recovered.strip(), False, {}
    parts = []
    parsed = {}
    for key in structured_keys:
        value = str(obj.get(key, "")).strip()
        parsed[key] = value
        if value:
            parts.append(value)
    return "\n".join(parts).strip(), True, parsed


def stage_evaluate(args: argparse.Namespace, manifest: dict, transcripts: dict, candidates: dict) -> dict:
    out_dir = resolve(args.output_dir)
    eval_path = out_dir / "evaluation.json"
    if eval_path.exists() and not args.overwrite:
        return load_json(eval_path)

    set_seed(args.seed)
    structured_keys = candidates["structured_keys"]
    candidate_by_id = {r["id"]: r for r in candidates["records"]}
    transcript_by_id = {r["id"]: r for r in transcripts["records"]}
    manifest_by_id = {r["id"]: r for r in manifest["sample"]}
    model = Model(args.victim_model, "4bit" if args.victim_quantize_4bit else None)

    rows = []
    for sample in manifest["sample"]:
        sample_id = sample["id"]
        print(f"Evaluating strict attack: {sample_id}", flush=True)
        result_dir = result_path(sample)
        original_prompt = (result_dir / "original_prompt.txt").read_text(encoding="utf-8").strip()
        transcript = transcript_by_id[sample_id]
        candidate_record = candidate_by_id[sample_id]
        selection_queries = [x["user_input"] for x in transcript["selection"]]
        selection_refs = [x["protected_output"] for x in transcript["selection"]]
        eval_queries = transcript["eval_inputs"]

        cand_rows = []
        for cand in candidate_record["candidates"]:
            prompt_text, parsed, parsed_fields = reconstruct_prompt(cand["recovered_prompt"], structured_keys)
            if selection_queries:
                selection_outputs = generate_plain_outputs(model, prompt_text, selection_queries, args)
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
        original_outputs = generate_plain_outputs(model, original_prompt, eval_queries, args)
        protected_outputs = generate_soft_outputs(model, result_dir, eval_queries, args)
        recovered_outputs = generate_plain_outputs(model, best["executable_recovered_prompt"], eval_queries, args)
        null_outputs = generate_plain_outputs(model, "", eval_queries, args)

        mismatch_prompt = ""
        if len(manifest["sample"]) > 1:
            other = manifest["sample"][(sample["sample_index"] + 1) % len(manifest["sample"])]
            other_result = result_path(other)
            mismatch_prompt = (other_result / "original_prompt.txt").read_text(encoding="utf-8").strip()
        mismatch_outputs = generate_plain_outputs(model, mismatch_prompt, eval_queries, args)

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
            "sample_index": sample["sample_index"],
            "id": sample_id,
            "category_for_manifest_only": manifest_by_id[sample_id]["category_for_manifest_only"],
            "result_dir": sample["result_dir"],
            "original_prompt": original_prompt,
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
        rows.append(row)

    summary = summarize(rows)
    payload = {
        "generation_config": generation_args(args),
        "records": rows,
        "summary": summary,
        "audit": audit_recovery_inputs(transcripts["records"]),
    }
    write_json(eval_path, payload)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", manifest, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def summarize(rows: list[dict]) -> dict:
    out = {
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
    denom = max(1e-9, out["mean_functional_f1_protected_vs_original"] - out["mean_mismatch_f1_vs_original"])
    out["normalized_recovery_gain_vs_mismatch"] = (
        out["mean_functional_f1_recovered_vs_original"] - out["mean_mismatch_f1_vs_original"]
    ) / denom
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category_for_manifest_only"]].append(row)
    for cat, cat_rows in by_cat.items():
        out["by_category"][cat] = {
            "n": len(cat_rows),
            "mean_functional_f1_recovered_vs_original": mean(r["mean_f1_recovered_vs_original"] for r in cat_rows),
            "mean_functional_f1_recovered_vs_protected": mean(r["mean_f1_recovered_vs_protected"] for r in cat_rows),
            "mean_text_f1_recovered_prompt_vs_original": mean(r["selected_text_f1_vs_original_prompt"] for r in cat_rows),
        }
    return out


def write_report(path: Path, manifest: dict, evaluation: dict) -> None:
    s = evaluation["summary"]
    audit = evaluation["audit"]
    lines = [
        "# Strict Black-Box Step 2 Report",
        "",
        "## Setting",
        "",
        "- Recovery input: user input + protected model output only.",
        "- Forbidden inputs: prompt category, prompt label, paper title, original prompt, soft prompt, projection text, scores, metadata.",
        f"- Prompts evaluated: {s['n_prompts']}",
        f"- Evidence / selection / eval queries: {manifest['evidence_queries']} / {manifest['selection_queries']} / {manifest['eval_queries']}",
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
        f"| Text F1: recovered prompt vs original prompt | {s['mean_text_f1_recovered_prompt_vs_original']:.3f} |",
        f"| Functional F1: recovered output vs original output | {s['mean_functional_f1_recovered_vs_original']:.3f} |",
        f"| Functional F1: recovered output vs protected output | {s['mean_functional_f1_recovered_vs_protected']:.3f} |",
        f"| Defense utility: protected output vs original output | {s['mean_functional_f1_protected_vs_original']:.3f} |",
        f"| Null baseline: null output vs original output | {s['mean_null_f1_vs_original']:.3f} |",
        f"| Mismatch baseline: wrong prompt output vs original output | {s['mean_mismatch_f1_vs_original']:.3f} |",
        f"| Normalized recovery gain vs mismatch | {s['normalized_recovery_gain_vs_mismatch']:.3f} |",
        f"| JSON parse rate | {s['json_parse_rate']:.3f} |",
        "",
    ]
    lines.extend(["", "## By Category", "", "| Category | N | Functional F1 vs original | Functional F1 vs protected | Text F1 |", "|---|---:|---:|---:|---:|"])
    for cat in CATEGORIES:
        if cat not in s["by_category"]:
            continue
        row = s["by_category"][cat]
        lines.append(
            f"| {cat} | {row['n']} | {row['mean_functional_f1_recovered_vs_original']:.3f} | "
            f"{row['mean_functional_f1_recovered_vs_protected']:.3f} | {row['mean_text_f1_recovered_prompt_vs_original']:.3f} |"
        )
    lines.extend(["", "## Selected Adapter Counts", "", json.dumps(s["selected_adapter_counts"], indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("results/strict_blackbox_step2_20"))
    parser.add_argument("--result_prefix", default="sixcat200")
    parser.add_argument("--recovery_model_tag", default="qwen3_32b")
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--category_quotas", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence_queries", type=int, default=6)
    parser.add_argument("--selection_queries", type=int, default=2)
    parser.add_argument("--eval_queries", type=int, default=10)
    parser.add_argument("--victim_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--victim_gpu", type=int, default=4)
    parser.add_argument("--victim_quantize_4bit", action="store_true", default=True)
    parser.add_argument("--victim_batch_size", type=int, default=4)
    parser.add_argument("--recovery_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--recovery_gpu", type=int, default=5)
    parser.add_argument("--recovery_max_new_tokens", type=int, default=220)
    parser.add_argument("--structured_keys", default=",".join(DEFAULT_STRUCTURED_KEYS))
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--stage", choices=["all", "manifest", "collect", "recover", "evaluate", "report"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_recovery_inputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = stage_manifest(args)
    if args.stage == "manifest":
        print(resolve(args.output_dir) / "manifest.json")
        return 0
    transcripts = stage_collect(args, manifest) if args.stage in {"all", "collect"} else load_json(out_dir / "evidence_transcripts.json")
    if args.stage == "collect":
        print(resolve(args.output_dir) / "evidence_transcripts.json")
        return 0
    candidates = stage_recover(args, transcripts) if args.stage in {"all", "recover"} else load_json(out_dir / "adapter_candidates.json")
    if args.stage == "recover":
        print(resolve(args.output_dir) / "adapter_candidates.json")
        return 0
    evaluation = stage_evaluate(args, manifest, transcripts, candidates) if args.stage in {"all", "evaluate"} else load_json(out_dir / "evaluation.json")
    if args.stage == "report":
        write_report(out_dir / "REPORT.md", manifest, evaluation)
    print(out_dir / "REPORT.md")
    print(json.dumps(evaluation["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
