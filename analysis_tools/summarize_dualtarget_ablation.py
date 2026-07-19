#!/usr/bin/env python3
"""Compare regex-v1, silver-v2, and dual-target recovery."""

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


def utility(path: Path) -> tuple[float, float]:
    summary = json.loads(
        (path / "recovery_utility_eval_matched_n6.json").read_text(encoding="utf-8")
    )["summary"]
    return summary["mean_prompt_token_f1"], summary["success_rates"]["success@0.50"]


def main() -> int:
    rows = []
    for slug, label in CATEGORIES:
        v1_path = BASE / f"results/sixcat150_{slug}_structured_qwen14b_lora"
        v2_path = BASE / f"results/sixcat150_{slug}_structured_qwen14b_lora_v2targets"
        dual_path = BASE / f"results/sixcat150_{slug}_structured_qwen14b_lora_dualtargets"
        dual_text = json.loads((dual_path / "lora_recovery_eval.json").read_text(encoding="utf-8"))["averages"]
        rows.append(
            {
                "category": label,
                "v1": utility(v1_path),
                "v2": utility(v2_path),
                "dual": utility(dual_path),
                "dual_exec_text_f1": dual_text["executable_token_f1"],
                "dual_structured_text_f1": dual_text["structured_token_f1"],
                "dual_json_valid": dual_text["dual_json_parse_success"],
            }
        )

    lines = [
        "# Dual-Target Recovery Ablation",
        "",
        "All runs use the same Qwen2.5-14B QLoRA configuration, seed-42 120/30 split, n=6 evidence, and matched probes.",
        "",
        "| Category | Utility v1 | Utility v2 | Utility dual | Dual vs v2 | Success dual | Executable Text F1 | Structured Text F1 | Valid JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['v1'][0]:.3f} | {row['v2'][0]:.3f} | "
            f"{row['dual'][0]:.3f} | {row['dual'][0]-row['v2'][0]:+.3f} | "
            f"{row['dual'][1]:.3f} | {row['dual_exec_text_f1']:.3f} | "
            f"{row['dual_structured_text_f1']:.3f} | {row['dual_json_valid']:.3f} |"
        )
    macro = {}
    for name, getter in {
        "v1_utility": lambda r: r["v1"][0],
        "v2_utility": lambda r: r["v2"][0],
        "dual_utility": lambda r: r["dual"][0],
        "dual_success": lambda r: r["dual"][1],
        "dual_exec_text_f1": lambda r: r["dual_exec_text_f1"],
        "dual_structured_text_f1": lambda r: r["dual_structured_text_f1"],
        "dual_json_valid": lambda r: r["dual_json_valid"],
    }.items():
        macro[name] = sum(getter(row) for row in rows) / len(rows)
    lines.append(
        f"| **Macro average** | {macro['v1_utility']:.3f} | {macro['v2_utility']:.3f} | "
        f"{macro['dual_utility']:.3f} | {macro['dual_utility']-macro['v2_utility']:+.3f} | "
        f"{macro['dual_success']:.3f} | {macro['dual_exec_text_f1']:.3f} | "
        f"{macro['dual_structured_text_f1']:.3f} | {macro['dual_json_valid']:.3f} |"
    )
    out = BASE / "results/sixcat150_dualtarget_ablation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "macro": macro}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
