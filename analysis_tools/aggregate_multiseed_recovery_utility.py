#!/usr/bin/env python3
"""Aggregate multi-seed recovery utility runs into a stability report.

For each (category, n) condition that has multiple seed files
(`recovery_utility_eval_seed*.json`), compute:
  - mean utility F1 across seeds (per prompt, then averaged)
  - per-prompt std across seeds (then averaged)
  - cross-seed success-rate variation
  - 95% CI for mean utility F1 (using std across seeds)

Then juxtapose against:
  - cross-n gap (how big is the difference between best and worst n)
  - is cross-n gap > 2 * mean per-prompt std? -> structural
                                 ~ similar     -> noise
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict


BASE = Path(__file__).resolve().parents[1]


def lora_dir_for(cat: str, n: int) -> Path:
    suffix = "" if n == 6 else f"_n{n}"
    return BASE / f"results/remaining5_{cat}_structured_qwen14b_lora{suffix}"


def load_seed_files(lora_dir: Path) -> dict[int, dict]:
    out = {}
    for f in lora_dir.glob("recovery_utility_eval_seed*.json"):
        stem = f.stem  # recovery_utility_eval_seed42
        if "seed" not in stem:
            continue
        try:
            seed = int(stem.rsplit("seed", 1)[-1])
        except ValueError:
            continue
        out[seed] = json.loads(f.read_text(encoding="utf-8"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", default="constraints,task_instruction")
    p.add_argument("--n_list", default="6,12,18,24,30")
    p.add_argument("--out_csv", type=Path, default=Path("results/recovery_utility_multiseed_summary.csv"))
    p.add_argument("--out_md", type=Path, default=Path("results/recovery_utility_multiseed_summary.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    ns = [int(x) for x in args.n_list.split(",") if x.strip()]

    rows = []
    per_prompt_dump = {}
    for cat in categories:
        for n in ns:
            lora_dir = lora_dir_for(cat, n)
            seed_data = load_seed_files(lora_dir)
            if not seed_data:
                continue
            seeds = sorted(seed_data)
            # Collect per-prompt utility F1 across seeds.
            prompt_scores: dict[str, dict[int, float]] = defaultdict(dict)
            for seed, payload in seed_data.items():
                for rec in payload.get("records", []):
                    prompt_scores[rec["id"]][seed] = float(rec["mean_pair_token_f1"])

            per_prompt_stats = []
            for pid, seed_to_score in prompt_scores.items():
                vals = [seed_to_score[s] for s in seeds if s in seed_to_score]
                if len(vals) < 2:
                    continue
                mean = sum(vals) / len(vals)
                std = statistics.pstdev(vals)
                per_prompt_stats.append((pid, mean, std, vals))

            if not per_prompt_stats:
                continue

            mean_of_means = sum(p[1] for p in per_prompt_stats) / len(per_prompt_stats)
            mean_of_stds = sum(p[2] for p in per_prompt_stats) / len(per_prompt_stats)
            std_of_means = statistics.pstdev([p[1] for p in per_prompt_stats]) if len(per_prompt_stats) > 1 else 0.0

            # Per-seed summary aggregates (success rates etc.)
            seed_means = []
            seed_succ_050 = []
            for seed in seeds:
                s = seed_data[seed]["summary"]
                seed_means.append(float(s["mean_prompt_token_f1"]))
                seed_succ_050.append(float(s["success_rates"].get("success@0.50", 0.0)))
            seed_mean_mean = sum(seed_means) / len(seed_means)
            seed_mean_std = statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0
            succ_050_mean = sum(seed_succ_050) / len(seed_succ_050)
            succ_050_std = statistics.pstdev(seed_succ_050) if len(seed_succ_050) > 1 else 0.0

            rows.append({
                "category": cat,
                "n": n,
                "n_prompts": len(per_prompt_stats),
                "n_seeds": len(seeds),
                "seeds": ",".join(str(s) for s in seeds),
                "mean_utility_f1": mean_of_means,
                "mean_per_prompt_std": mean_of_stds,
                "std_of_per_prompt_means": std_of_means,
                "seed_mean_of_means": seed_mean_mean,
                "seed_std_of_means": seed_mean_std,
                "succ_050_mean": succ_050_mean,
                "succ_050_std": succ_050_std,
            })
            per_prompt_dump[f"{cat}_n{n}"] = [
                {"id": p[0], "mean": p[1], "std": p[2], "values": p[3]}
                for p in per_prompt_stats
            ]

    if not rows:
        print("No multi-seed files found.")
        return 1

    out_csv = args.out_csv if args.out_csv.is_absolute() else BASE / args.out_csv
    out_md = args.out_md if args.out_md.is_absolute() else BASE / args.out_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Cross-n analysis per category: how big is the n-axis variation
    # compared to the per-prompt seed-axis variation?
    cat_analysis = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        if len(cat_rows) < 2:
            continue
        means = [r["mean_utility_f1"] for r in cat_rows]
        cross_n_range = max(means) - min(means)
        cross_n_std = statistics.pstdev(means)
        avg_per_prompt_std = sum(r["mean_per_prompt_std"] for r in cat_rows) / len(cat_rows)
        cat_analysis[cat] = {
            "cross_n_range": cross_n_range,
            "cross_n_std": cross_n_std,
            "avg_per_prompt_std": avg_per_prompt_std,
            "ratio_range_over_std": (cross_n_range / avg_per_prompt_std) if avg_per_prompt_std > 0 else float("inf"),
        }

    lines = ["# Multi-Seed Recovery-Utility Stability Report", ""]
    lines.append("Per-seed evaluations of `eval_recovery_utility.py` with sampling")
    lines.append("seeds {42, 123, 2025}. The `mean_per_prompt_std` column is the average")
    lines.append("across held-out prompts of the std of utility Token F1 over seeds.")
    lines.append("")
    lines.append("## Per-condition stability")
    lines.append("")
    lines.append("| Category | n | n_prompts | seeds | Mean Utility F1 | Mean per-prompt std | Succ@0.50 (mean ± std) |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['n']} | {row['n_prompts']} | "
            f"{row['seeds']} | "
            f"{row['mean_utility_f1']:.3f} | "
            f"{row['mean_per_prompt_std']:.3f} | "
            f"{row['succ_050_mean']:.3f} ± {row['succ_050_std']:.3f} |"
        )
    lines.append("")

    lines.append("## Cross-n vs cross-seed comparison (structural-vs-noise check)")
    lines.append("")
    lines.append("`cross_n_range` is the spread of `mean_utility_f1` across the n values for the same category.")
    lines.append("`avg_per_prompt_std` is the average within-prompt std across seeds.")
    lines.append("If `ratio = cross_n_range / avg_per_prompt_std` is well above 2, the n-axis")
    lines.append("differences are larger than the seed-noise floor and the n-sweep pattern")
    lines.append("is structural. If it is close to or below 1, the n-sweep pattern is")
    lines.append("indistinguishable from seed noise.")
    lines.append("")
    lines.append("| Category | cross_n_range | avg_per_prompt_std | ratio |")
    lines.append("|---|---:|---:|---:|")
    for cat, a in cat_analysis.items():
        lines.append(f"| {cat} | {a['cross_n_range']:.3f} | {a['avg_per_prompt_std']:.3f} | {a['ratio_range_over_std']:.2f} |")
    lines.append("")

    # Spot-check per-prompt variance for the most extreme conditions
    lines.append("## Per-prompt detail (most-extreme condition per category)")
    lines.append("")
    for cat in categories:
        cat_keys = [k for k in per_prompt_dump if k.startswith(f"{cat}_n")]
        if not cat_keys:
            continue
        # Pick the condition with largest mean std (most variance)
        def cond_std(k):
            data = per_prompt_dump[k]
            return sum(d["std"] for d in data) / len(data) if data else 0
        worst = max(cat_keys, key=cond_std)
        data = per_prompt_dump[worst]
        lines.append(f"### {worst} (per-prompt across seeds)")
        lines.append("")
        lines.append("| Prompt id | mean | std | values |")
        lines.append("|---|---:|---:|---|")
        for d in sorted(data, key=lambda x: -x["std"]):
            vals_str = ", ".join(f"{v:.3f}" for v in d["values"])
            lines.append(f"| {d['id']} | {d['mean']:.3f} | {d['std']:.3f} | {vals_str} |")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
