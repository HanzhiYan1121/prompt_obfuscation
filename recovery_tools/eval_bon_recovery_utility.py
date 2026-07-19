#!/usr/bin/env python3
"""Best-of-N reranking evaluation of recovered prompts.

Procedure (per held-out prompt):
  1. Load N candidate recovered prompts from
     `lora_recovered_candidates_N{N}.json` (produced by
     `generate_recovery_candidates.py`).
  2. Split the 16 cached test queries into two halves:
       indices [0 .. rerank_split-1]   -> rerank validation set
       indices [rerank_split .. 15]    -> final evaluation set
  3. For each candidate, reconstruct prompt text, run the model on the
     rerank set, compute mean per-pair Token F1 against the cached
     conventional outputs for that half.
  4. Pick the candidate with the highest rerank score; in ties, pick the
     first.
  5. Run the picked candidate on the final eval set and report:
       - per-prompt utility F1 on the final eval set
       - aggregate mean utility F1 and Success@tau

We also report a baseline using the first candidate (== greedy-equivalent
when sampling with low temperature) evaluated on the same final eval set,
so the comparison is apples-to-apples.

Avoids leakage: rerank and final eval sets are disjoint halves of the same
test query split. The candidate that wins reranking has not seen the final
eval queries during reranking; it only inherits its prompt-text identity.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader

from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses
from src.utils import set_seed


BASE = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURED_KEYS = ["role", "task", "format", "constraints", "context", "tool_protocol"]
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)
WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


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


def flatten(group) -> str:
    return "\n".join(str(x) for x in group) if isinstance(group, list) else str(group)


def parse_structured(text: str, keys: list[str]) -> dict[str, str] | None:
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
                    cand = text[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return {k: str(obj.get(k, "")) for k in keys}
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def reconstruct(parsed: dict[str, str], keys: list[str]) -> str:
    parts = []
    for k in keys:
        v = (parsed.get(k) or "").strip()
        if v:
            parts.append(v)
    return "\n".join(parts).strip()


def resolve_recovered_text(raw: str, keys: list[str]) -> tuple[str, bool]:
    parsed = parse_structured(raw, keys)
    if parsed is not None:
        return reconstruct(parsed, keys), False
    return (raw or "").strip(), True


def gen(model: Model, system_prompt: str, queries: list[str],
        batch_size: int, gen_args: dict) -> list[str]:
    if not queries:
        return []
    dataset = TextDataset(queries)
    collate_fn = create_collate_fn(tokenizer=model.tokenizer, system_prompt=system_prompt or "")
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    raw = generate_model_responses(model, dl, gen_args)
    return [flatten(g) for g in raw]


def aggregate_success(scores: list[float], thresholds) -> dict[str, float]:
    if not scores:
        return {f"success@{t:.2f}": 0.0 for t in thresholds}
    n = len(scores)
    return {f"success@{t:.2f}": sum(1 for s in scores if s >= t) / n for t in thresholds}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", type=Path, required=True)
    p.add_argument("--inputs_path", type=Path, required=True)
    p.add_argument("--candidates_name", default="lora_recovered_candidates_N5.json")
    p.add_argument("--n_queries", type=int, default=16)
    p.add_argument("--rerank_split", type=int, default=8,
                   help="First K test queries used for reranking; remaining used for final eval.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--quantize_4bit", action="store_true", default=True)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--num_return_sequences", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--structured_keys", default=",".join(DEFAULT_STRUCTURED_KEYS))
    p.add_argument("--thresholds", default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS))
    p.add_argument("--output_name", default="recovery_utility_bon_eval.json")
    return p.parse_args()


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else BASE / p


def index_inputs(p: Path) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def main() -> int:
    args = parse_args()
    lora_dir = resolve(args.lora_dir)
    inputs_path = resolve(args.inputs_path)
    candidates_path = lora_dir / args.candidates_name
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}")
    structured_keys = [k.strip() for k in args.structured_keys.split(",") if k.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    set_seed(args.seed)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cand_records = json.loads(candidates_path.read_text(encoding="utf-8"))
    inputs_index = index_inputs(inputs_path)

    gen_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
    }

    quant = "4bit" if args.quantize_4bit else None
    model = Model(args.model_name, quant)

    out_records = []
    for i, cand_rec in enumerate(cand_records):
        rid = cand_rec["id"]
        if rid not in inputs_index:
            print(f"[{i+1}/{len(cand_records)}] {rid}: missing inputs entry, skip")
            continue
        result_dir = resolve(Path(inputs_index[rid]["result_dir"]))
        try:
            queries_all = json.loads((result_dir / "prepared_data" / "test_data.json").read_text(encoding="utf-8"))[:args.n_queries]
            conv_all = json.loads((result_dir / "conventional_output.json").read_text(encoding="utf-8"))["output"][:args.n_queries]
            conv_all = [flatten(g) for g in conv_all]
        except FileNotFoundError as e:
            print(f"[{i+1}/{len(cand_records)}] {rid}: cache missing ({e}), skip")
            continue
        n = min(len(queries_all), len(conv_all))
        queries_all = queries_all[:n]
        conv_all = conv_all[:n]
        if n <= args.rerank_split:
            print(f"[{i+1}/{len(cand_records)}] {rid}: not enough queries (n={n}), skip")
            continue

        rerank_queries = queries_all[: args.rerank_split]
        eval_queries = queries_all[args.rerank_split:]
        rerank_conv = conv_all[: args.rerank_split]
        eval_conv = conv_all[args.rerank_split:]

        # Reconstruct each candidate; score on rerank half.
        candidate_scores = []
        per_candidate = []
        for j, raw in enumerate(cand_rec["candidates"]):
            cand_text, fallback = resolve_recovered_text(raw, structured_keys)
            if not cand_text:
                candidate_scores.append(0.0)
                per_candidate.append({
                    "index": j, "rerank_score": 0.0, "eval_score": None,
                    "recovered_text": "", "fallback": fallback,
                })
                continue
            rerank_outs = gen(model, cand_text, rerank_queries, args.eval_batch_size, gen_args)
            rerank_scores = [token_f1(rerank_outs[k], rerank_conv[k]) for k in range(len(rerank_queries))]
            r_mean = sum(rerank_scores) / len(rerank_scores)
            candidate_scores.append(r_mean)
            per_candidate.append({
                "index": j, "rerank_score": r_mean,
                "rerank_per_pair": rerank_scores,
                "recovered_text": cand_text,
                "fallback": fallback,
            })

        # Pick winner; evaluate on final half.
        winner_idx = max(range(len(candidate_scores)), key=lambda k: candidate_scores[k])
        winner_text = per_candidate[winner_idx]["recovered_text"]
        if winner_text:
            eval_outs = gen(model, winner_text, eval_queries, args.eval_batch_size, gen_args)
            eval_scores = [token_f1(eval_outs[k], eval_conv[k]) for k in range(len(eval_queries))]
        else:
            eval_outs = ["" for _ in eval_queries]
            eval_scores = [0.0 for _ in eval_queries]
        winner_eval_mean = sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
        per_candidate[winner_idx]["eval_score"] = winner_eval_mean
        per_candidate[winner_idx]["eval_per_pair"] = eval_scores

        # Baseline: first candidate (analogous to greedy/seed=42 single pick)
        baseline_idx = 0
        baseline_text = per_candidate[baseline_idx]["recovered_text"]
        if baseline_idx != winner_idx and baseline_text:
            base_eval_outs = gen(model, baseline_text, eval_queries, args.eval_batch_size, gen_args)
            base_eval_scores = [token_f1(base_eval_outs[k], eval_conv[k]) for k in range(len(eval_queries))]
        elif baseline_idx == winner_idx:
            base_eval_scores = eval_scores
        else:
            base_eval_outs = ["" for _ in eval_queries]
            base_eval_scores = [0.0 for _ in eval_queries]
        baseline_eval_mean = sum(base_eval_scores) / len(base_eval_scores) if base_eval_scores else 0.0
        per_candidate[baseline_idx]["baseline_eval_score"] = baseline_eval_mean
        per_candidate[baseline_idx]["baseline_eval_per_pair"] = base_eval_scores

        out_records.append({
            "id": rid,
            "result_dir": str(result_dir.relative_to(BASE)) if result_dir.is_relative_to(BASE) else str(result_dir),
            "n_candidates": len(cand_rec["candidates"]),
            "rerank_split": args.rerank_split,
            "n_eval_pairs": len(eval_queries),
            "winner_idx": winner_idx,
            "bon_eval_f1": winner_eval_mean,
            "baseline_eval_f1": baseline_eval_mean,
            "improvement": winner_eval_mean - baseline_eval_mean,
            "candidate_rerank_scores": candidate_scores,
            "per_candidate": per_candidate,
        })
        print(f"[{i+1}/{len(cand_records)}] {rid}: "
              f"winner=#{winner_idx} (rerank={candidate_scores[winner_idx]:.3f}); "
              f"baseline_eval={baseline_eval_mean:.3f}, bon_eval={winner_eval_mean:.3f}, "
              f"delta={winner_eval_mean - baseline_eval_mean:+.3f}")

    if out_records:
        bon_scores = [r["bon_eval_f1"] for r in out_records]
        base_scores = [r["baseline_eval_f1"] for r in out_records]
        summary = {
            "n_prompts": len(out_records),
            "mean_baseline_eval_f1": sum(base_scores) / len(base_scores),
            "mean_bon_eval_f1": sum(bon_scores) / len(bon_scores),
            "mean_improvement": (sum(bon_scores) - sum(base_scores)) / len(bon_scores),
            "baseline_success_rates": aggregate_success(base_scores, thresholds),
            "bon_success_rates": aggregate_success(bon_scores, thresholds),
            "rerank_split": args.rerank_split,
            "thresholds": thresholds,
        }
    else:
        summary = {"n_prompts": 0}

    out_path = lora_dir / args.output_name
    out_path.write_text(json.dumps({"summary": summary, "records": out_records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Saved {out_path}")
    if out_records:
        print(f"  n_prompts = {summary['n_prompts']}")
        print(f"  baseline mean F1 = {summary['mean_baseline_eval_f1']:.3f}")
        print(f"  BoN      mean F1 = {summary['mean_bon_eval_f1']:.3f}")
        print(f"  mean improvement = {summary['mean_improvement']:+.3f}")
        print("  baseline success: " + ", ".join(f"{k}={v:.3f}" for k, v in summary["baseline_success_rates"].items()))
        print("  BoN      success: " + ", ".join(f"{k}={v:.3f}" for k, v in summary["bon_success_rates"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
