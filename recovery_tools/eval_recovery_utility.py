#!/usr/bin/env python3
"""Utility-based recovery evaluation.

For each held-out original prompt, run the model with the recovered prompt
as system prompt on the same test queries that were used to build the
conventional cached outputs, then compare the resulting outputs to the
conventional outputs. Recovery is judged successful when the per-prompt
behavior similarity exceeds a calibrated threshold.

This addresses SY's request for an actual "successful recovery" definition
with real TP/FP/FN, instead of just text-overlap scores between the
recovered and the original prompt text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses
from src.utils import set_seed


BASE = Path(__file__).resolve().parents[1]

DEFAULT_STRUCTURED_KEYS = ["role", "task", "format", "constraints", "context", "tool_protocol"]
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)


# -------------------- text + scoring helpers --------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def token_f1(pred: str, ref: str) -> float:
    pc = Counter(_tokens(pred))
    rc = Counter(_tokens(ref))
    if not pc or not rc:
        return 0.0
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pc.values())
    recall = overlap / sum(rc.values())
    return 2 * precision * recall / (precision + recall)


def flatten_output_group(group) -> str:
    if isinstance(group, list):
        return "\n".join(str(x) for x in group)
    return str(group)


# -------------------- recovered-prompt reconstruction --------------------

def _parse_structured_recovered(text: str, keys: list[str]) -> dict[str, str] | None:
    """Try to parse a JSON object containing the structured 6D fields.

    The recovery model sometimes emits trailing junk after the JSON. We accept
    the first {...} block that parses.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return {k: str(obj.get(k, "")) for k in keys}
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def reconstruct_prompt_from_structured(parsed: dict[str, str], keys: list[str]) -> str:
    """Concatenate non-empty 6D fields into a plain-text system prompt.

    Field order matches the taxonomy. Empty fields are skipped so the model
    is not told "constraints: <empty>" which would itself be a hint.
    """
    executable = (parsed.get("executable_prompt") or "").strip()
    if executable:
        return executable

    parts = []
    for key in keys:
        if key == "executable_prompt":
            continue
        value = (parsed.get(key) or "").strip()
        if not value:
            continue
        # The role/task/etc are content-rich enough that we drop the label
        # prefix to mirror how the original prompts were written. Keeping the
        # label adds tokens the original did not have and depresses token-F1
        # comparisons unfairly.
        parts.append(value)
    return "\n".join(parts).strip()


def resolve_recovered_prompt_text(record: dict, structured_keys: list[str]) -> tuple[str, dict]:
    """Return (reconstructed_prompt_text, debug_info)."""
    recovered_raw = record.get("recovered_prompt", "") or ""
    parsed = _parse_structured_recovered(recovered_raw, structured_keys)
    if parsed is not None:
        reconstructed = reconstruct_prompt_from_structured(parsed, structured_keys)
        return reconstructed, {"parsed_structured": parsed, "fallback": False}
    # Fallback: treat recovered output as free text prompt.
    return recovered_raw.strip(), {"parsed_structured": None, "fallback": True}


# -------------------- generation --------------------

def generate_with_prompt(
    model: Model,
    system_prompt: str,
    test_queries: list[str],
    batch_size: int,
    generation_args: dict,
) -> list[str]:
    if not system_prompt:
        # An empty system prompt is meaningful for the random/null baseline.
        system_prompt = ""
    dataset = TextDataset(test_queries)
    collate_fn = create_collate_fn(tokenizer=model.tokenizer, system_prompt=system_prompt)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    raw = generate_model_responses(model, dataloader, generation_args)
    # generate_model_responses returns list[list[str]] of size n_queries x num_return_sequences.
    return [flatten_output_group(group) for group in raw]


# -------------------- main pipeline --------------------

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def load_test_data(
    result_dir: Path,
    limit: int,
    conventional_output_name: str = "conventional_output.json",
    offset: int = 0,
) -> list[str]:
    if conventional_output_name != "conventional_output.json":
        payload = json.loads((result_dir / conventional_output_name).read_text(encoding="utf-8"))
        return [str(x) for x in payload["input"][offset : offset + limit]]
    payload = json.loads((result_dir / "prepared_data" / "test_data.json").read_text(encoding="utf-8"))
    return [str(x) for x in payload[offset : offset + limit]]


def load_conventional_outputs(
    result_dir: Path,
    limit: int,
    conventional_output_name: str = "conventional_output.json",
    offset: int = 0,
) -> list[str]:
    payload = json.loads((result_dir / conventional_output_name).read_text(encoding="utf-8"))
    return [
        flatten_output_group(group)
        for group in payload["output"][offset : offset + limit]
    ]


def index_recovery_inputs(inputs_path: Path) -> dict[str, dict]:
    records = json.loads(inputs_path.read_text(encoding="utf-8"))
    return {r["id"]: r for r in records}


def aggregate_success(prompt_scores: list[float], thresholds: Iterable[float]) -> dict[str, float]:
    if not prompt_scores:
        return {f"success@{t:.2f}": 0.0 for t in thresholds}
    out = {}
    n = len(prompt_scores)
    for t in thresholds:
        out[f"success@{t:.2f}"] = sum(1 for s in prompt_scores if s >= t) / n
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", type=Path, required=True,
                        help="Directory containing lora_recovered_prompts.json.")
    parser.add_argument("--inputs_path", type=Path, required=True,
                        help="recovery_inputs.json used during recovery training/eval.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--quantize_4bit", action="store_true", default=True)
    parser.add_argument("--no_quantize", dest="quantize_4bit", action="store_false")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--n_queries", type=int, default=16,
                        help="How many test queries per prompt to evaluate on.")
    parser.add_argument(
        "--query_offset",
        type=int,
        default=0,
        help="Start index in the held-out query list; use this to exclude recovery-evidence queries.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Use 1 by default for speed; existing conventional cache used 1 per index.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structured_keys", default=",".join(DEFAULT_STRUCTURED_KEYS))
    parser.add_argument("--thresholds", default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS))
    parser.add_argument("--output_name", default="recovery_utility_eval.json",
                        help="Output filename written inside --lora_dir.")
    parser.add_argument("--conventional_output_name", default="conventional_output.json",
                        help="Conventional output cache to compare against. If not the default, queries are read from this cache's input field.")
    parser.add_argument("--random_baseline", action="store_true",
                        help="Also compute random-prompt lower bound using shuffled originals.")
    parser.add_argument("--self_similarity", action="store_true",
                        help="Also compute self-similarity upper bound by re-running the original prompt.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip prompts already present in an existing output file.")
    parser.add_argument("--max_records", type=int, default=0,
                        help="If >0, evaluate at most this many records (for smoke tests).")
    return parser.parse_args()


def resolve_under_base(p: Path) -> Path:
    return p if p.is_absolute() else BASE / p


def main() -> int:
    args = parse_args()
    lora_dir = resolve_under_base(args.lora_dir)
    inputs_path = resolve_under_base(args.inputs_path)
    recovered_path = lora_dir / "lora_recovered_prompts.json"
    if not recovered_path.exists():
        raise FileNotFoundError(f"Expected recovered prompts at {recovered_path}")

    structured_keys = [k.strip() for k in args.structured_keys.split(",") if k.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    set_seed(args.seed)

    recovered_records = json.loads(recovered_path.read_text(encoding="utf-8"))
    if args.max_records > 0:
        recovered_records = recovered_records[: args.max_records]
    inputs_index = index_recovery_inputs(inputs_path)

    output_path = lora_dir / args.output_name
    existing = {}
    if args.skip_existing and output_path.exists():
        prev = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {row["id"]: row for row in prev.get("records", [])}

    # Lazy-load model only when we have at least one prompt to evaluate.
    model: Model | None = None
    generation_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
    }

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_records: list[dict] = []
    random_records: list[dict] = []
    self_records: list[dict] = []

    # Cache random pool of recovered text reconstructions so the random baseline
    # is sampled from the same distribution as the recovered prompts.
    all_recovered_texts: list[str] = []
    for record in recovered_records:
        text, _ = resolve_recovered_prompt_text(record, structured_keys)
        if text:
            all_recovered_texts.append(text)

    rng = random.Random(args.seed)

    for i, record in enumerate(recovered_records):
        rec_id = record["id"]
        if rec_id in existing:
            out_records.append(existing[rec_id])
            print(f"[{i+1}/{len(recovered_records)}] skip cached {rec_id}")
            continue
        if rec_id not in inputs_index:
            print(f"[{i+1}/{len(recovered_records)}] missing inputs entry: {rec_id}")
            continue
        input_entry = inputs_index[rec_id]
        result_dir = resolve_under_base(Path(input_entry["result_dir"]))
        if not result_dir.exists():
            print(f"[{i+1}/{len(recovered_records)}] missing result_dir: {result_dir}")
            continue

        try:
            test_queries = load_test_data(
                result_dir,
                args.n_queries,
                args.conventional_output_name,
                args.query_offset,
            )
            conv_outputs = load_conventional_outputs(
                result_dir,
                args.n_queries,
                args.conventional_output_name,
                args.query_offset,
            )
        except FileNotFoundError as exc:
            print(f"[{i+1}/{len(recovered_records)}] data missing for {rec_id}: {exc}")
            continue

        n = min(len(test_queries), len(conv_outputs))
        test_queries = test_queries[:n]
        conv_outputs = conv_outputs[:n]
        if n == 0:
            print(f"[{i+1}/{len(recovered_records)}] no eval pairs for {rec_id}")
            continue

        recovered_text, debug = resolve_recovered_prompt_text(record, structured_keys)
        if not recovered_text:
            print(f"[{i+1}/{len(recovered_records)}] empty recovered text for {rec_id}")
            row = {
                "id": rec_id,
                "result_dir": str(result_dir.relative_to(BASE)),
                "n_pairs": n,
                "mean_pair_token_f1": 0.0,
                "per_pair_token_f1": [0.0] * n,
                "recovered_prompt_text": "",
                "fallback_used": debug["fallback"],
            }
            out_records.append(row)
            continue

        if model is None:
            quant = "4bit" if args.quantize_4bit else None
            model = Model(args.model_name, quant)

        recovered_outputs = generate_with_prompt(
            model, recovered_text, test_queries, args.eval_batch_size, generation_args
        )

        pair_scores = [token_f1(recovered_outputs[j], conv_outputs[j]) for j in range(n)]
        mean_score = sum(pair_scores) / n

        out_records.append(
            {
                "id": rec_id,
                "result_dir": str(result_dir.relative_to(BASE)),
                "n_pairs": n,
                "mean_pair_token_f1": mean_score,
                "per_pair_token_f1": pair_scores,
                "recovered_prompt_text": recovered_text,
                "recovered_outputs": recovered_outputs,
                "conventional_outputs": conv_outputs,
                "fallback_used": debug["fallback"],
            }
        )
        print(
            f"[{i+1}/{len(recovered_records)}] {rec_id}: mean_token_f1={mean_score:.3f} "
            f"({n} pairs, fallback={debug['fallback']})"
        )

        if args.random_baseline and all_recovered_texts:
            # Pick a recovered-prompt text from a *different* original prompt
            # to simulate "what if the recovery output were unrelated to this id".
            candidates = [t for t in all_recovered_texts if t != recovered_text]
            if candidates:
                random_text = rng.choice(candidates)
                random_outputs = generate_with_prompt(
                    model, random_text, test_queries, args.eval_batch_size, generation_args
                )
                random_pair = [token_f1(random_outputs[j], conv_outputs[j]) for j in range(n)]
                random_records.append(
                    {
                        "id": rec_id,
                        "mean_pair_token_f1": sum(random_pair) / n,
                        "per_pair_token_f1": random_pair,
                    }
                )

        if args.self_similarity:
            original_prompt_text = (result_dir / "original_prompt.txt").read_text(encoding="utf-8").strip()
            # Re-run the original prompt with a *different* seed inside generation,
            # so we measure the floor of sampling noise. set_seed reseeds globals.
            set_seed(args.seed + 1)
            self_outputs = generate_with_prompt(
                model, original_prompt_text, test_queries, args.eval_batch_size, generation_args
            )
            set_seed(args.seed)
            self_pair = [token_f1(self_outputs[j], conv_outputs[j]) for j in range(n)]
            self_records.append(
                {
                    "id": rec_id,
                    "mean_pair_token_f1": sum(self_pair) / n,
                    "per_pair_token_f1": self_pair,
                }
            )

    prompt_scores = [row["mean_pair_token_f1"] for row in out_records if "mean_pair_token_f1" in row]
    success_rates = aggregate_success(prompt_scores, thresholds)

    summary = {
        "lora_dir": str(lora_dir.relative_to(BASE)) if lora_dir.is_relative_to(BASE) else str(lora_dir),
        "inputs_path": str(inputs_path.relative_to(BASE)) if inputs_path.is_relative_to(BASE) else str(inputs_path),
        "n_prompts": len(prompt_scores),
        "mean_prompt_token_f1": (sum(prompt_scores) / len(prompt_scores)) if prompt_scores else 0.0,
        "thresholds": thresholds,
        "success_rates": success_rates,
        "generation_args": generation_args,
        "model_name": args.model_name,
        "n_queries_per_prompt": args.n_queries,
        "query_offset": args.query_offset,
        "conventional_output_name": args.conventional_output_name,
    }

    if random_records:
        rand_scores = [r["mean_pair_token_f1"] for r in random_records]
        summary["random_baseline"] = {
            "n_prompts": len(rand_scores),
            "mean_prompt_token_f1": sum(rand_scores) / len(rand_scores),
            "success_rates": aggregate_success(rand_scores, thresholds),
        }
    if self_records:
        self_scores = [r["mean_pair_token_f1"] for r in self_records]
        summary["self_similarity_ceiling"] = {
            "n_prompts": len(self_scores),
            "mean_prompt_token_f1": sum(self_scores) / len(self_scores),
            "success_rates": aggregate_success(self_scores, thresholds),
        }

    payload = {
        "summary": summary,
        "records": out_records,
    }
    if random_records:
        payload["random_records"] = random_records
    if self_records:
        payload["self_similarity_records"] = self_records

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Saved {output_path.relative_to(BASE)}")
    print(f"  n_prompts = {summary['n_prompts']}")
    print(f"  mean_prompt_token_f1 = {summary['mean_prompt_token_f1']:.3f}")
    for k, v in summary["success_rates"].items():
        print(f"  {k} = {v:.3f}")
    if "random_baseline" in summary:
        print("  random_baseline:")
        print(f"    mean = {summary['random_baseline']['mean_prompt_token_f1']:.3f}")
        for k, v in summary["random_baseline"]["success_rates"].items():
            print(f"    {k} = {v:.3f}")
    if "self_similarity_ceiling" in summary:
        print("  self_similarity_ceiling:")
        print(f"    mean = {summary['self_similarity_ceiling']['mean_prompt_token_f1']:.3f}")
        for k, v in summary["self_similarity_ceiling"]["success_rates"].items():
            print(f"    {k} = {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
