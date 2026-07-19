#!/usr/bin/env python3
"""Summarize per-category tau calibration outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CATEGORIES = [
    ("role_persona", "Role/Persona"),
    ("task_instruction", "Task Instruction"),
    ("format_spec", "Format Spec"),
    ("constraints", "Constraints"),
    ("context_knowledge", "Context Knowledge"),
    ("tool_protocol", "Tool/Protocol"),
]


def success(summary: dict, block: str, tau: float) -> float:
    target = f"success@{tau:.2f}"
    if block == "recovered":
        return float(summary["success_rates"].get(target, 0.0))
    return float(summary.get(block, {}).get("success_rates", {}).get(target, 0.0))


def mean(summary: dict, block: str) -> float:
    if block == "recovered":
        return float(summary.get("mean_prompt_token_f1", 0.0))
    return float(summary.get(block, {}).get("mean_prompt_token_f1", 0.0))


def choose_tau(summary: dict, thresholds: list[float]) -> tuple[float | None, str]:
    """Choose a per-category operating tau.

    Conservative rule:
    - random success <= 0.10,
    - self-similarity success >= 0.20,
    - recovered success > random success,
    - recovered success > 0.

    If no threshold satisfies all conditions, report no calibrated tau rather
    than selecting a threshold that no recovered prompt can pass.
    """
    usable = []
    for tau in thresholds:
        rand = success(summary, "random_baseline", tau)
        self_s = success(summary, "self_similarity_ceiling", tau)
        rec = success(summary, "recovered", tau)
        if rand <= 0.10 and self_s >= 0.20 and rec > rand and rec > 0:
            usable.append((tau, rec, rand, self_s))
    if usable:
        tau, rec, rand, self_s = max(usable, key=lambda x: (x[0], x[1] - x[2]))
        return tau, f"random<=0.10, self>=0.20, recovered>random; rec={rec:.3f}, rand={rand:.3f}, self={self_s:.3f}"
    return None, "no threshold satisfies random<=0.10, self>=0.20, and recovered>random"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_prefix", default="sixcat150")
    ap.add_argument("--run_suffix", default="")
    ap.add_argument("--input_name", default="recovery_utility_tau_calibration_matched_n6.json")
    ap.add_argument("--out_md", default="results/sixcat150_tau_calibration_summary.md")
    ap.add_argument("--out_json", default="results/sixcat150_tau_calibration_summary.json")
    args = ap.parse_args()

    rows = []
    for slug, label in CATEGORIES:
        p = Path("results") / f"{args.result_prefix}_{slug}_structured_qwen14b_lora{args.run_suffix}" / args.input_name
        if not p.exists():
            rows.append({"category": label, "missing": str(p)})
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        s = payload["summary"]
        thresholds = [float(x) for x in s["thresholds"]]
        tau, reason = choose_tau(s, thresholds)
        row = {
            "category": label,
            "n_prompts": s["n_prompts"],
            "recovered_mean": mean(s, "recovered"),
            "random_mean": mean(s, "random_baseline"),
            "self_mean": mean(s, "self_similarity_ceiling"),
            "recommended_tau": tau,
            "reason": reason,
        }
        for t in thresholds:
            row[f"rec@{t:.2f}"] = success(s, "recovered", t)
            row[f"rand@{t:.2f}"] = success(s, "random_baseline", t)
            row[f"self@{t:.2f}"] = success(s, "self_similarity_ceiling", t)
        rows.append(row)

    Path(args.out_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Six-Category 150 Per-Category Tau Calibration",
        "",
        "Calibration uses matched original-test probes with random-prompt floor and self-similarity ceiling.",
        "",
        "| Category | Rec mean | Random mean | Self mean | Recommended tau | Reason |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if "missing" in row:
            lines.append(f"| {row['category']} | n/a | n/a | n/a | n/a | missing `{row['missing']}` |")
            continue
        lines.append(
            f"| {row['category']} | {row['recovered_mean']:.3f} | {row['random_mean']:.3f} | "
            f"{row['self_mean']:.3f} | "
            f"{row['recommended_tau']:.2f} | {row['reason']} |" if row["recommended_tau"] is not None else
            f"| {row['category']} | {row['recovered_mean']:.3f} | {row['random_mean']:.3f} | "
            f"{row['self_mean']:.3f} | n/a | {row['reason']} |"
        )

    lines += [
        "",
        "## Full Success Rates",
        "",
    ]
    for row in rows:
        if "missing" in row:
            continue
        thresholds = sorted(float(k.split("@")[1]) for k in row if k.startswith("rec@"))
        lines.append(f"### {row['category']}")
        lines.append("")
        lines.append("| Tau | Recovered | Random floor | Self-sim ceiling |")
        lines.append("|---:|---:|---:|---:|")
        for t in thresholds:
            lines.append(f"| {t:.2f} | {row[f'rec@{t:.2f}']:.3f} | {row[f'rand@{t:.2f}']:.3f} | {row[f'self@{t:.2f}']:.3f} |")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.out_md}")
    print(f"Saved {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
