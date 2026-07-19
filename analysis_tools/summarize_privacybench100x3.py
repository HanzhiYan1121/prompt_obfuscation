#!/usr/bin/env python3
"""Summarize the three additional privacy-defense recovery benchmarks."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

METHODS = [
    ("emojicrypt", "EmojiCrypt / EmojiPrompt-style"),
    ("dp_opt", "DP-OPT-style"),
    ("preempt", "Prεεmpt-style"),
]

CATEGORIES = [
    ("role_persona", "Role/Persona"),
    ("task_instruction", "Task Instruction"),
    ("format_spec", "Format Spec"),
    ("constraints", "Constraints"),
    ("context_knowledge", "Context Knowledge"),
    ("tool_protocol", "Tool/Protocol"),
]

SOURCES = {
    "EmojiCrypt / EmojiPrompt-style": "https://arxiv.org/abs/2402.05868",
    "DP-OPT-style": "https://arxiv.org/abs/2312.03724",
    "Prεεmpt-style": "https://www.ndss-symposium.org/ndss-paper/pr%CE%B5%CE%B5mpt-sanitizing-sensitive-prompts-for-llms/",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_text_eval(path: Path) -> dict:
    return json.loads((path / "lora_recovery_eval.json").read_text(encoding="utf-8"))


def load_utility_eval(path: Path) -> dict:
    return json.loads((path / "recovery_utility_eval_matched_n6.json").read_text(encoding="utf-8"))


def result_dir(method: str, category: str) -> Path:
    return BASE / f"results/privacybench100x3_{method}_{category}_structured_qwen3_32b_lora"


def main() -> int:
    rows = []
    for method_slug, method_label in METHODS:
        for cat_slug, cat_label in CATEGORIES:
            d = result_dir(method_slug, cat_slug)
            text_eval = load_text_eval(d)["averages"]
            utility_summary = load_utility_eval(d)["summary"]
            rows.append(
                {
                    "method": method_label,
                    "method_slug": method_slug,
                    "category": cat_label,
                    "category_slug": cat_slug,
                    "n_prompts": utility_summary["n_prompts"],
                    "text_token_f1": text_eval["token_f1"],
                    "rougeL_f1": text_eval["rougeL_f1"],
                    "keyword_recall": text_eval.get("keyword_recall"),
                    "utility_f1": utility_summary["mean_prompt_token_f1"],
                    "success@0.30": utility_summary["success_rates"]["success@0.30"],
                    "success@0.50": utility_summary["success_rates"]["success@0.50"],
                    "success@0.70": utility_summary["success_rates"]["success@0.70"],
                }
            )

    by_method = []
    for _, method_label in METHODS:
        subset = [r for r in rows if r["method"] == method_label]
        by_method.append(
            {
                "method": method_label,
                "source": SOURCES[method_label],
                "text_token_f1": mean([r["text_token_f1"] for r in subset]),
                "rougeL_f1": mean([r["rougeL_f1"] for r in subset]),
                "utility_f1": mean([r["utility_f1"] for r in subset]),
                "success@0.50": mean([r["success@0.50"] for r in subset]),
            }
        )

    by_category = []
    for _, cat_label in CATEGORIES:
        subset = [r for r in rows if r["category"] == cat_label]
        by_category.append(
            {
                "category": cat_label,
                "text_token_f1": mean([r["text_token_f1"] for r in subset]),
                "rougeL_f1": mean([r["rougeL_f1"] for r in subset]),
                "utility_f1": mean([r["utility_f1"] for r in subset]),
                "success@0.50": mean([r["success@0.50"] for r in subset]),
            }
        )

    out_dir = BASE / "results/privacybench100x3_3defense_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "These are recovery-pipeline-compatible benchmark approximations of the "
            "three protection methods, not full reimplementations of each paper."
        ),
        "sources": SOURCES,
        "rows": rows,
        "by_method": by_method,
        "by_category": by_category,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Three Additional Prompt-Protection Benchmarks",
        "",
        "Important caveat: these runs use benchmark approximations aligned with our recovery pipeline, not full reproductions of each defense system.",
        "",
        "## Sources",
        "",
    ]
    for item in by_method:
        lines.append(f"- {item['method']}: {item['source']}")
    lines.extend(
        [
            "",
            "## Average by Protection Method",
            "",
            "| Method | Text F1 | ROUGE-L F1 | Utility F1 | Success@0.50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in by_method:
        lines.append(
            f"| {item['method']} | {item['text_token_f1']:.3f} | {item['rougeL_f1']:.3f} | "
            f"{item['utility_f1']:.3f} | {item['success@0.50']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Average by Prompt Category",
            "",
            "| Category | Text F1 | ROUGE-L F1 | Utility F1 | Success@0.50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in by_category:
        lines.append(
            f"| {item['category']} | {item['text_token_f1']:.3f} | {item['rougeL_f1']:.3f} | "
            f"{item['utility_f1']:.3f} | {item['success@0.50']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Full Matrix",
            "",
            "| Method | Category | Text F1 | ROUGE-L F1 | Utility F1 | Success@0.50 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['category']} | {r['text_token_f1']:.3f} | "
            f"{r['rougeL_f1']:.3f} | {r['utility_f1']:.3f} | {r['success@0.50']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
