#!/usr/bin/env python3
"""Aggregate parallel Role adaptive-probe pilot shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            ROOT / "results/role_adaptive_probe_pilot_part0/results.json",
            ROOT / "results/role_adaptive_probe_pilot_part1/results.json",
        ],
    )
    p.add_argument("--output_dir", type=Path, default=ROOT / "results/role_adaptive_probe_pilot")
    p.add_argument("--bootstrap_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def bootstrap_ci(differences: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(differences) for _ in differences]
        means.append(sum(draw) / len(draw))
    return [percentile(means, 0.025), percentile(means, 0.975)]


def main() -> int:
    args = parse_args()
    records = []
    configs = []
    for path in args.inputs:
        path = path if path.is_absolute() else ROOT / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["records"])
        configs.append(payload["config"])
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate prompt IDs across shards")
    methods = ["greedy", "random", "adaptive", "oracle"]
    summary = {"n_prompts": len(records)}
    for method in methods:
        scores = [row["final_utility"][method] for row in records]
        summary[method] = {
            "mean_utility_f1": sum(scores) / len(scores),
            "success@0.50": sum(score >= 0.5 for score in scores) / len(scores),
        }

    comparisons = {}
    for left, right in [("adaptive", "greedy"), ("random", "greedy"), ("adaptive", "random")]:
        differences = [
            row["final_utility"][left] - row["final_utility"][right] for row in records
        ]
        comparisons[f"{left}_minus_{right}"] = {
            "mean_difference": sum(differences) / len(differences),
            "bootstrap_95_ci": bootstrap_ci(
                differences, args.bootstrap_samples, args.seed + len(comparisons)
            ),
            "wins": sum(value > 1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
            "losses": sum(value < -1e-12 for value in differences),
        }
    summary["comparisons"] = comparisons
    summary["adaptive_matches_oracle_utility"] = sum(
        abs(row["final_utility"]["adaptive"] - row["final_utility"]["oracle"]) <= 1e-12
        for row in records
    ) / len(records)
    summary["random_matches_oracle_utility"] = sum(
        abs(row["final_utility"]["random"] - row["final_utility"]["oracle"]) <= 1e-12
        for row in records
    ) / len(records)

    payload = {"summary": summary, "configs": configs, "records": records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Role/Persona Adaptive-Probe Recovery Pilot",
        "",
        "## Controlled setting",
        "",
        "- 10 held-out Role/Persona prompts.",
        "- Four recovery candidates per prompt: one greedy candidate plus three sampled candidates.",
        "- Twelve candidate-conditioned role probes per prompt.",
        "- Four real queries to the obfuscated soft prompt for Random and Adaptive selection.",
        "- Final Utility F1 uses original-test queries 6--11. These queries are not among the first six behavior examples used for recovery.",
        "- Candidate and probe generation do not receive the original prompt. The evaluator uses the original prompt only after selection to generate final reference outputs.",
        "",
        "## Main results",
        "",
        "| Method | Mean Utility F1 | Success@0.50 |",
        "|---|---:|---:|",
    ]
    for method in methods:
        row = summary[method]
        lines.append(f"| {method.title()} | {row['mean_utility_f1']:.3f} | {row['success@0.50']:.3f} |")
    lines.extend(["", "| Comparison | Mean difference | Bootstrap 95% CI | Wins / ties / losses |", "|---|---:|---:|---:|"])
    for name, row in comparisons.items():
        ci = row["bootstrap_95_ci"]
        lines.append(
            f"| {name.replace('_', ' ')} | {row['mean_difference']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['wins']} / {row['ties']} / {row['losses']} |"
        )
    lines.extend(
        [
            "",
            f"- Adaptive matches oracle Utility on {summary['adaptive_matches_oracle_utility']:.1%} of prompts.",
            f"- Random matches oracle Utility on {summary['random_matches_oracle_utility']:.1%} of prompts.",
            "- Interpretation: candidate diversity provides the main gain over greedy recovery. The current adaptive disagreement rule is only slightly better than random reranking and its confidence interval includes zero.",
            "",
            "## Per-prompt results",
            "",
            "| Prompt | Greedy | Random | Adaptive | Oracle | Adaptive - Random |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in records:
        score = row["final_utility"]
        lines.append(
            f"| {row['id']} | {score['greedy']:.3f} | {score['random']:.3f} | "
            f"{score['adaptive']:.3f} | {score['oracle']:.3f} | "
            f"{score['adaptive'] - score['random']:+.3f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
