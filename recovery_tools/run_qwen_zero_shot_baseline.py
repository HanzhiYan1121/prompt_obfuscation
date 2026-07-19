#!/usr/bin/env python3
"""Qwen3-32B zero-shot recovery baseline for strict Step 2.

The baseline uses the same black-box transcripts as the strict ensemble attack,
but it loads only the base Qwen3-32B model and no recovery LoRA adapters.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "recovery_tools"))

from recovery_tools.run_strict_blackbox_step2 import (
    DEFAULT_STRUCTURED_KEYS,
    audit_recovery_inputs,
    build_blackbox_recovery_prompt,
    generate_plain_outputs,
    generate_recovery_text,
    load_recovery_base,
    mean,
    reconstruct_prompt,
    resolve,
    result_path,
    token_f1,
    write_json,
)
from recovery_tools.train_acl100_recovery_lora import parse_structured_keys
from src.model import Model
from src.utils import set_seed


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def generation_config(args: argparse.Namespace) -> dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": 1,
    }


def recover_zero_shot(args: argparse.Namespace) -> dict:
    out_dir = resolve(args.output_dir)
    out_path = out_dir / "qwen3_zero_shot_recovered_prompts.json"
    if out_path.exists() and not args.overwrite:
        return load_json(out_path)

    transcripts = load_json(out_dir / "evidence_transcripts.json")
    structured_keys = parse_structured_keys(args.structured_keys)
    tokenizer, model = load_recovery_base(args)

    rows = []
    for record in transcripts["records"]:
        print(f"Zero-shot recovering: {record['id']}", flush=True)
        prompt = build_blackbox_recovery_prompt(record, structured_keys)
        recovered = generate_recovery_text(tokenizer, model, prompt, args, structured_keys)
        rows.append(
            {
                "sample_index": record["sample_index"],
                "id": record["id"],
                "recovered_prompt": recovered,
            }
        )

    payload = {
        "method": "Qwen3-32B zero-shot recovery",
        "model": args.recovery_model,
        "uses_lora_adapter": False,
        "structured_keys": structured_keys,
        "audit": audit_recovery_inputs(transcripts["records"]),
        "records": rows,
    }
    write_json(out_path, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def evaluate_zero_shot(args: argparse.Namespace) -> dict:
    out_dir = resolve(args.output_dir)
    out_path = out_dir / "qwen3_zero_shot_evaluation.json"
    if out_path.exists() and not args.overwrite:
        return load_json(out_path)

    recovered_payload = recover_zero_shot(args)
    main_eval = load_json(out_dir / "evaluation.json")
    manifest = load_json(out_dir / "manifest.json")
    main_by_id = {r["id"]: r for r in main_eval["records"]}
    manifest_by_id = {r["id"]: r for r in manifest["sample"]}
    structured_keys = recovered_payload["structured_keys"]

    set_seed(args.seed)
    model = Model(args.victim_model, "4bit" if args.victim_quantize_4bit else None)
    rows = []
    for rec in recovered_payload["records"]:
        sample_id = rec["id"]
        print(f"Evaluating zero-shot baseline: {sample_id}", flush=True)
        main_row = main_by_id[sample_id]
        sample = manifest_by_id[sample_id]
        result_dir = result_path(sample)
        original_prompt = (result_dir / "original_prompt.txt").read_text(encoding="utf-8").strip()
        prompt_text, parsed, parsed_fields = reconstruct_prompt(rec["recovered_prompt"], structured_keys)
        eval_inputs = [q["user_input"] for q in main_row["eval"]]
        original_outputs = [q["original_output"] for q in main_row["eval"]]
        protected_outputs = [q["protected_output"] for q in main_row["eval"]]
        recovered_outputs = generate_plain_outputs(model, prompt_text, eval_inputs, args)

        per_query = []
        for idx, query in enumerate(eval_inputs):
            per_query.append(
                {
                    "query_index": idx,
                    "user_input": query,
                    "original_output": original_outputs[idx],
                    "protected_output": protected_outputs[idx],
                    "zero_shot_output": recovered_outputs[idx],
                    "f1_zero_shot_vs_original": token_f1(recovered_outputs[idx], original_outputs[idx]),
                    "f1_zero_shot_vs_protected": token_f1(recovered_outputs[idx], protected_outputs[idx]),
                }
            )

        rows.append(
            {
                "sample_index": rec["sample_index"],
                "id": sample_id,
                "category_for_manifest_only": sample["category_for_manifest_only"],
                "raw_recovered_prompt": rec["recovered_prompt"],
                "executable_recovered_prompt": prompt_text,
                "json_parse_success": parsed,
                "parsed_fields": parsed_fields,
                "text_f1_vs_original_prompt": token_f1(prompt_text, original_prompt),
                "mean_f1_zero_shot_vs_original": mean(q["f1_zero_shot_vs_original"] for q in per_query),
                "mean_f1_zero_shot_vs_protected": mean(q["f1_zero_shot_vs_protected"] for q in per_query),
                "eval": per_query,
            }
        )

    summary = {
        "method": "Qwen3-32B zero-shot recovery",
        "n_prompts": len(rows),
        "mean_utility_f1_vs_original": mean(r["mean_f1_zero_shot_vs_original"] for r in rows),
        "mean_utility_f1_vs_protected": mean(r["mean_f1_zero_shot_vs_protected"] for r in rows),
        "mean_text_f1_vs_original_prompt": mean(r["text_f1_vs_original_prompt"] for r in rows),
        "json_parse_rate": mean(1.0 if r["json_parse_success"] else 0.0 for r in rows),
    }
    payload = {
        "generation_config": generation_config(args),
        "summary": summary,
        "records": rows,
    }
    write_json(out_path, payload)
    write_json(out_dir / "qwen3_zero_shot_summary.json", summary)
    write_report(out_dir / "qwen3_zero_shot_REPORT.md", payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    lines = [
        "# Qwen3-32B Zero-Shot Recovery Baseline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Prompts | {summary['n_prompts']} |",
        f"| Utility F1 vs original | {summary['mean_utility_f1_vs_original']:.3f} |",
        f"| Utility F1 vs protected | {summary['mean_utility_f1_vs_protected']:.3f} |",
        f"| Text F1 vs original prompt | {summary['mean_text_f1_vs_original_prompt']:.3f} |",
        f"| JSON parse rate | {summary['json_parse_rate']:.3f} |",
        "",
        "| # | ID | Category | Utility F1 | Protected F1 | Text F1 | JSON |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["records"]:
        lines.append(
            f"| {row['sample_index'] + 1} | {row['id']} | {row['category_for_manifest_only']} | "
            f"{row['mean_f1_zero_shot_vs_original']:.3f} | {row['mean_f1_zero_shot_vs_protected']:.3f} | "
            f"{row['text_f1_vs_original_prompt']:.3f} | {int(row['json_parse_success'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("results/strict_blackbox_step2_20"))
    parser.add_argument("--stage", choices=["all", "recover", "evaluate"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--recovery_gpu", type=int, default=5)
    parser.add_argument("--recovery_max_new_tokens", type=int, default=220)
    parser.add_argument("--structured_keys", default=",".join(DEFAULT_STRUCTURED_KEYS))
    parser.add_argument("--victim_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--victim_gpu", type=int, default=4)
    parser.add_argument("--victim_quantize_4bit", action="store_true", default=True)
    parser.add_argument("--victim_batch_size", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage in {"all", "recover"}:
        recover_zero_shot(args)
        if args.stage == "recover":
            print(resolve(args.output_dir) / "qwen3_zero_shot_recovered_prompts.json")
            return 0
    payload = evaluate_zero_shot(args)
    print(resolve(args.output_dir) / "qwen3_zero_shot_REPORT.md")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
