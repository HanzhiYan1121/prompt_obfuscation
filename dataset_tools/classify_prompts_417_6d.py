#!/usr/bin/env python3
"""Classify prompts_417.json with the existing six-dimension taxonomy."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classify_allprev_prompts_6d import DIMENSIONS, DISPLAY, assign_category, score_prompt


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "system prompt" / "prompts_417.json"
OUT_DIR = ROOT / "results" / "prompts_417_6d_classification"


def load_records(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, payload
    return payload, payload.get("records", [])


def prompt_text(rec: dict) -> str:
    return rec.get("text") or rec.get("prompt_text") or rec.get("prompt") or ""


def prompt_label(rec: dict) -> str:
    return rec.get("prompt_name") or rec.get("prompt_label") or rec.get("prompt_category") or ""


def paper_title(rec: dict) -> str:
    return rec.get("title") or rec.get("paper_title") or rec.get("paper_file") or ""


def record_id(rec: dict, idx: int) -> str:
    if rec.get("global_id") is not None:
        return f"Prompt417-{int(rec['global_id']):03d}"
    if rec.get("id"):
        return str(rec["id"])
    return f"Prompt417-{idx + 1:03d}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata, records = load_records(PROMPTS_PATH)

    rows = []
    for idx, rec in enumerate(records):
        text = prompt_text(rec)
        scores = score_prompt(text, prompt_label(rec), paper_title(rec))
        category = assign_category(scores)
        row = {
            "id": record_id(rec, idx),
            "global_id": rec.get("global_id", idx + 1),
            "paper_file": rec.get("paper_file", ""),
            "title": paper_title(rec),
            "venue": rec.get("venue", ""),
            "prompt_name": prompt_label(rec),
            "source_group": rec.get("source_group", ""),
            "source_prompt_file": rec.get("source_prompt_file", ""),
            "original_prompt_category": rec.get("prompt_category", ""),
            "category": category,
            **scores,
            "prompt_chars": len(text),
            "prompt_words": len(re.findall(r"\w+", text)),
            "text_sha256_16": rec.get("text_sha256_16", ""),
            "text": text,
        }
        rows.append(row)

    fieldnames = [
        "id",
        "global_id",
        "paper_file",
        "title",
        "venue",
        "prompt_name",
        "source_group",
        "source_prompt_file",
        "original_prompt_category",
        "category",
        *DIMENSIONS,
        "prompt_chars",
        "prompt_words",
        "text_sha256_16",
        "text",
    ]
    with (OUT_DIR / "prompts_417_6d_classification.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "prompts_417_6d_classification.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    grouped = {}
    for cat, items in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        grouped[cat] = {
            "n": len(items),
            "avg_scores": {d: sum(x[d] for x in items) / len(items) for d in DIMENSIONS},
            "avg_prompt_words": sum(x["prompt_words"] for x in items) / len(items),
        }
    (OUT_DIR / "prompts_417_6d_group_summary.json").write_text(
        json.dumps(grouped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cats = list(grouped)
    counts = [grouped[c]["n"] for c in cats]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.bar(cats, counts, color="#4e79a7")
    ax.set_title(f"Prompts by 6D-Derived Category (n={len(rows)})")
    ax.set_ylabel("Prompt count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prompts_417_6d_category_counts.png", dpi=220)
    plt.close(fig)

    heat = np.array([[grouped[c]["avg_scores"][d] for d in DIMENSIONS] for c in cats])
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
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
    fig.savefig(OUT_DIR / "prompts_417_6d_category_heatmap.png", dpi=220)
    plt.close(fig)

    avg_scores = [sum(row[d] for row in rows) / len(rows) for d in DIMENSIONS]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    values = avg_scores + avg_scores[:1]
    radar_angles = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(7.4, 6.4), subplot_kw={"polar": True})
    ax.plot(radar_angles, values, color="#4e79a7", linewidth=2.5)
    ax.fill(radar_angles, values, color="#4e79a7", alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels([DISPLAY[d] for d in DIMENSIONS], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#777777")
    ax.set_title(f"Average 6D Profile (n={len(rows)})", fontsize=14, fontweight="bold", pad=20)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prompts_417_6d_average_radar.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Prompts 417 6D Classification",
        "",
        f"- File: `{PROMPTS_PATH.relative_to(ROOT)}`",
        f"- Metadata total_prompt_records: `{metadata.get('total_prompt_records', 'N/A')}`",
        f"- Actual records classified: `{len(rows)}`",
        "",
        "| Category | N | Avg words | Role | Task | Format | Constraints | Context | Tool |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat in cats:
        g = grouped[cat]
        s = g["avg_scores"]
        lines.append(
            f"| {cat} | {g['n']} | {g['avg_prompt_words']:.1f} | "
            f"{s['Role_Persona']:.2f} | {s['Task_Instruction']:.2f} | {s['Format_Specification']:.2f} | "
            f"{s['Constraints_Guardrails']:.2f} | {s['Context_Knowledge']:.2f} | {s['Tool_Action_Protocol']:.2f} |"
        )
    (OUT_DIR / "prompts_417_6d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT_DIR / "prompts_417_6d_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
