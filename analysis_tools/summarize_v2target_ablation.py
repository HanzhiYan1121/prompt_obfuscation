#!/usr/bin/env python3
"""Summarize controlled v1-regex vs v2-silver target recovery results."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CATEGORIES = [
    ("role_persona", "Role/Persona"),
    ("task_instruction", "Task Instruction"),
    ("format_spec", "Format Spec"),
    ("constraints", "Constraints"),
    ("context_knowledge", "Context Knowledge"),
    ("tool_protocol", "Tool/Protocol"),
]


def load_metrics(path: Path) -> tuple[float, float, float]:
    text = json.loads((path / "lora_recovery_eval.json").read_text(encoding="utf-8"))
    utility = json.loads(
        (path / "recovery_utility_eval_matched_n6.json").read_text(encoding="utf-8")
    )["summary"]
    return (
        float(text["averages"]["token_f1"]),
        float(utility["mean_prompt_token_f1"]),
        float(utility["success_rates"]["success@0.50"]),
    )


def main() -> int:
    rows = []
    for slug, label in CATEGORIES:
        old = load_metrics(BASE / f"results/sixcat150_{slug}_structured_qwen14b_lora")
        new = load_metrics(BASE / f"results/sixcat150_{slug}_structured_qwen14b_lora_v2targets")
        rows.append({"slug": slug, "category": label, "v1": old, "v2": new})

    lines = [
        "# Structured Target Ablation: Regex v1 vs Silver v2",
        "",
        "All conditions use Qwen2.5-14B QLoRA, seed 42, the same 120/30 split, n=6 recovery evidence, and matched original-test probes. Only the target construction changes.",
        "",
        "| Category | Text F1 v1 | Text F1 v2 | Delta | Utility F1 v1 | Utility F1 v2 | Delta | Success@0.50 v1 | Success@0.50 v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    payload = []
    for row in rows:
        t1, u1, s1 = row["v1"]
        t2, u2, s2 = row["v2"]
        lines.append(
            f"| {row['category']} | {t1:.3f} | {t2:.3f} | {t2-t1:+.3f} | "
            f"{u1:.3f} | {u2:.3f} | {u2-u1:+.3f} | {s1:.3f} | {s2:.3f} |"
        )
        payload.append(
            {
                "category": row["category"],
                "v1": {"text_f1": t1, "utility_f1": u1, "success_at_0_50": s1},
                "v2": {"text_f1": t2, "utility_f1": u2, "success_at_0_50": s2},
                "delta": {"text_f1": t2 - t1, "utility_f1": u2 - u1, "success_at_0_50": s2 - s1},
            }
        )
    macro_v1 = [sum(row["v1"][i] for row in rows) / len(rows) for i in range(3)]
    macro_v2 = [sum(row["v2"][i] for row in rows) / len(rows) for i in range(3)]
    lines.extend(
        [
            f"| **Macro average** | {macro_v1[0]:.3f} | {macro_v2[0]:.3f} | {macro_v2[0]-macro_v1[0]:+.3f} | {macro_v1[1]:.3f} | {macro_v2[1]:.3f} | {macro_v2[1]-macro_v1[1]:+.3f} | {macro_v1[2]:.3f} | {macro_v2[2]:.3f} |",
            "",
        ]
    )
    out_dir = BASE / "results/sixcat150_structured_targets_v2_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"categories": payload, "macro_v1": macro_v1, "macro_v2": macro_v2},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
