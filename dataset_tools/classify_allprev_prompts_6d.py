#!/usr/bin/env python3
"""Classify all_previous prompts with the existing 6-dimension taxonomy."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "system prompt" / "all_previous_system_prompts_runner.json"
OUT_DIR = ROOT / "results" / "allprev_6d_classification"
UTILITY_PATH = ROOT / "results" / "allprev_utility" / "allprev_utility_metrics.json"
RECOVERY_PATH = ROOT / "results" / "allprev_recovery" / "recovery_eval.json"
LORA_PATH = ROOT / "results" / "allprev_recovery_lora" / "lora_recovery_eval.json"

DIMENSIONS = [
    "Role_Persona",
    "Task_Instruction",
    "Format_Specification",
    "Constraints_Guardrails",
    "Context_Knowledge",
    "Tool_Action_Protocol",
]

DISPLAY = {
    "Role_Persona": "Role/Persona",
    "Task_Instruction": "Task Instruction",
    "Format_Specification": "Format Spec",
    "Constraints_Guardrails": "Constraints",
    "Context_Knowledge": "Context Knowledge",
    "Tool_Action_Protocol": "Tool/Protocol",
}


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def count_any(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


def cap_score(raw: float) -> float:
    return float(max(0, min(5, round(raw * 2) / 2)))


def score_prompt(prompt: str, label: str = "", paper_title: str = "") -> dict[str, float]:
    text = f"{label}\n{paper_title}\n{prompt}"
    lower = text.lower()
    words = re.findall(r"\w+", lower)
    length = len(words)

    role_hits = count_any(
        text,
        [
            r"\byou are\b",
            r"\bact as\b",
            r"\bimagine you are\b",
            r"\bpersona\b",
            r"\brole\b",
            r"\bassistant\b",
            r"\bexpert\b",
            r"\badvisor\b",
            r"\bagent\b",
            r"\btutor\b",
            r"\bjudge\b",
        ],
    )
    task_hits = count_any(
        text,
        [
            r"\byour task\b",
            r"\btask:\b",
            r"\binstruction\b",
            r"\bgenerate\b",
            r"\bclassify\b",
            r"\bevaluate\b",
            r"\banswer\b",
            r"\bidentify\b",
            r"\bextract\b",
            r"\bsummarize\b",
            r"\brank\b",
            r"\btranslate\b",
            r"\bsolve\b",
            r"\bwrite\b",
            r"\bcreate\b",
        ],
    )
    format_hits = count_any(
        text,
        [
            r"\bformat\b",
            r"\boutput\b",
            r"\bjson\b",
            r"\byaml\b",
            r"\bxml\b",
            r"\bcsv\b",
            r"\bschema\b",
            r"\btemplate\b",
            r"\bquery:\b",
            r"\banswer:\b",
            r"\[end\]",
            r"\[stop\]",
            r"<system>|</system>|<user>|</user>",
            r"\blist\b",
            r"\btable\b",
            r"\breturn\b",
        ],
    )
    constraint_hits = count_any(
        text,
        [
            r"\bmust\b",
            r"\bshould\b",
            r"\bdo not\b",
            r"\bdon't\b",
            r"\bnever\b",
            r"\bavoid\b",
            r"\bensure\b",
            r"\bonly\b",
            r"\bwithout\b",
            r"\bconfidential\b",
            r"\bsafety\b",
            r"\bguardrail\b",
            r"\bmalicious\b",
            r"\bdangerous\b",
        ],
    )
    context_hits = count_any(
        text,
        [
            r"\bprovided\b",
            r"\bgiven\b",
            r"\bpaper\b",
            r"\babstract\b",
            r"\bcontext\b",
            r"\bbackground\b",
            r"\bdocument\b",
            r"\bimage\b",
            r"\btable\b",
            r"\bexample\b",
            r"\bdemonstration\b",
            r"\bcategory\b",
            r"\bdomain\b",
            r"\bknowledge\b",
        ],
    )
    tool_hits = count_any(
        text,
        [
            r"\btool\b",
            r"\bapi\b",
            r"\bfunction\b",
            r"\bcode\b",
            r"\bpython\b",
            r"\bexecute\b",
            r"\bcall\b",
            r"\bsearch\b",
            r"\bbrowse\b",
            r"\baction\b",
            r"\bagent\b",
            r"\bstep[- ]?by[- ]?step\b",
            r"\bworkflow\b",
            r"\bprotocol\b",
        ],
    )

    # Length gives extra task/context/constraint signal because long prompts often
    # contain explicit instructions and background material.
    length_bonus = min(1.0, math.log(max(length, 1), 250)) if length > 25 else 0.0

    scores = {
        "Role_Persona": cap_score(0.8 * role_hits + (1.0 if re.search(r"\byou are\b|\bact as\b", lower) else 0.0)),
        "Task_Instruction": cap_score(0.55 * task_hits + 1.2 + 0.5 * length_bonus),
        "Format_Specification": cap_score(0.65 * format_hits + (1.0 if re.search(r"json|format|output|template|query:|answer:", lower) else 0.0)),
        "Constraints_Guardrails": cap_score(0.45 * constraint_hits + (1.0 if re.search(r"do not|must|never|avoid|only", lower) else 0.0)),
        "Context_Knowledge": cap_score(0.5 * context_hits + 0.6 * length_bonus),
        "Tool_Action_Protocol": cap_score(0.8 * tool_hits + (1.0 if re.search(r"tool|api|function|code|agent", lower) else 0.0)),
    }
    return scores


def assign_category(scores: dict[str, float]) -> str:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], DIMENSIONS.index(kv[0])))
    top_dim, top_score = ordered[0]
    second_score = ordered[1][1]
    active = [d for d, v in scores.items() if v >= 3.5]
    if len(active) >= 3:
        return "Mixed/Complex"
    if top_score - second_score < 0.5 and top_score >= 3:
        return "Mixed/Complex"
    return DISPLAY[top_dim]


def index_records(path: Path, score_key: str) -> dict[str, dict]:
    payload = load_json(path, {})
    return {r["id"]: r for r in payload.get("records", [])}


def mean_or_none(items: list[dict], key: str) -> float | None:
    vals = [x[key] for x in items if key in x]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_json(PROMPTS_PATH, [])
    utility = index_records(UTILITY_PATH, "utility_preservation_score")
    baseline = index_records(RECOVERY_PATH, "token_f1")
    lora = index_records(LORA_PATH, "token_f1")

    rows = []
    for rec in prompts:
        scores = score_prompt(rec["prompt_text"], rec.get("prompt_label", ""), rec.get("paper_title", ""))
        category = assign_category(scores)
        row = {
            "id": rec["id"],
            "prompt_label": rec.get("prompt_label", ""),
            "paper_title": rec.get("paper_title", ""),
            "category": category,
            **scores,
            "prompt_chars": len(rec.get("prompt_text", "")),
            "prompt_words": len(re.findall(r"\w+", rec.get("prompt_text", ""))),
        }
        if rec["id"] in utility:
            row["utility_preservation_score"] = utility[rec["id"]]["utility_preservation_score"]
        if rec["id"] in baseline:
            row["baseline_recovery_token_f1"] = baseline[rec["id"]]["token_f1"]
            row["baseline_recovery_rougeL_f1"] = baseline[rec["id"]]["rougeL_f1"]
        if rec["id"] in lora:
            row["lora_recovery_token_f1"] = lora[rec["id"]]["token_f1"]
            row["lora_recovery_rougeL_f1"] = lora[rec["id"]]["rougeL_f1"]
        rows.append(row)

    fieldnames = [
        "id",
        "prompt_label",
        "paper_title",
        "category",
        *DIMENSIONS,
        "prompt_chars",
        "prompt_words",
        "utility_preservation_score",
        "baseline_recovery_token_f1",
        "baseline_recovery_rougeL_f1",
        "lora_recovery_token_f1",
        "lora_recovery_rougeL_f1",
    ]
    with (OUT_DIR / "allprev_6d_classification.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "allprev_6d_classification.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    grouped = {}
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    for cat, items in sorted(by_cat.items()):
        grouped[cat] = {
            "n": len(items),
            "avg_scores": {d: sum(x[d] for x in items) / len(items) for d in DIMENSIONS},
            "utility_n": sum("utility_preservation_score" in x for x in items),
            "baseline_n": sum("baseline_recovery_token_f1" in x for x in items),
            "lora_n": sum("lora_recovery_token_f1" in x for x in items),
            "avg_utility": mean_or_none(items, "utility_preservation_score"),
            "avg_baseline_token_f1": mean_or_none(items, "baseline_recovery_token_f1"),
            "avg_lora_token_f1": mean_or_none(items, "lora_recovery_token_f1"),
        }
    (OUT_DIR / "allprev_6d_group_summary.json").write_text(json.dumps(grouped, indent=2, ensure_ascii=False), encoding="utf-8")
    cats = list(grouped)

    # Overall 6D radar plot.
    avg_scores = [sum(row[d] for row in rows) / len(rows) for d in DIMENSIONS]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    radar_values = avg_scores + avg_scores[:1]
    radar_angles = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(7.4, 6.4), subplot_kw={"polar": True})
    ax.plot(radar_angles, radar_values, color="#4e79a7", linewidth=2.5)
    ax.fill(radar_angles, radar_values, color="#4e79a7", alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels([DISPLAY[d] for d in DIMENSIONS], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#777777")
    ax.set_title("Average 6-Dimension Profile — 101 System Prompts", fontsize=14, fontweight="bold", pad=20)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "allprev_6d_average_radar.png", dpi=220)
    plt.close(fig)

    # Category-by-dimension heatmap.
    heat = np.array([[grouped[c]["avg_scores"][d] for d in DIMENSIONS] for c in cats])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(heat, cmap="YlGnBu", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(np.arange(len(DIMENSIONS)))
    ax.set_xticklabels([DISPLAY[d] for d in DIMENSIONS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_title("Average 6-Dimension Scores by Prompt Category", fontsize=13, fontweight="bold")
    for i in range(len(cats)):
        for j in range(len(DIMENSIONS)):
            ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#1f2933")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average dimension score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "allprev_6d_category_heatmap.png", dpi=220)
    plt.close(fig)

    # Category count plot.
    counts = [grouped[c]["n"] for c in cats]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(cats, counts, color="#4e79a7")
    ax.set_title("All Previous Prompts by 6-Dimension Category")
    ax.set_ylabel("Prompt count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "allprev_6d_category_counts.png", dpi=220)
    plt.close(fig)

    # Category metric plot.
    metric_cats = cats
    utility_vals = [grouped[c]["avg_utility"] or 0 for c in metric_cats]
    baseline_vals = [grouped[c]["avg_baseline_token_f1"] or 0 for c in metric_cats]
    lora_vals = [grouped[c]["avg_lora_token_f1"] or 0 for c in metric_cats]
    lora_missing = [grouped[c]["avg_lora_token_f1"] is None for c in metric_cats]
    x = np.arange(len(metric_cats))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width, utility_vals, width, label="Utility", color="#4c9c70")
    ax.bar(x, baseline_vals, width, label="Baseline recovery Token F1", color="#dd8452")
    ax.bar(x + width, lora_vals, width, label="QLoRA recovery Token F1 (held-out only)", color="#c14b52")
    for xpos, is_missing in zip(x + width, lora_missing):
        if is_missing:
            ax.text(xpos, 0.025, "N/A", ha="center", va="bottom", fontsize=8, rotation=90, color="#666666")
    ax.set_ylim(0, max(0.7, max(utility_vals + baseline_vals + lora_vals) + 0.08))
    ax.set_title("Utility and Recovery by Prompt Category")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cats, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "allprev_6d_category_metrics.png", dpi=220)
    plt.close(fig)

    lines = ["# All Previous Prompts 6D Classification", ""]
    lines.append("| Category | N | Utility | Baseline Token F1 | QLoRA Token F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in cats:
        g = grouped[cat]
        utility_text = "N/A" if g["avg_utility"] is None else f"{g['avg_utility']:.3f}"
        baseline_text = "N/A" if g["avg_baseline_token_f1"] is None else f"{g['avg_baseline_token_f1']:.3f}"
        lora_text = "N/A" if g["avg_lora_token_f1"] is None else f"{g['avg_lora_token_f1']:.3f}"
        lines.append(f"| {cat} | {g['n']} | {utility_text} | {baseline_text} | {lora_text} |")
    (OUT_DIR / "allprev_6d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT_DIR / "allprev_6d_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
