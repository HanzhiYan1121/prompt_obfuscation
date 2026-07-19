#!/usr/bin/env python3
"""Prepare runner files for the five non-role 6D prompt categories.

For each category, use all prompts if the category has <= max_per_category
examples across all sources. If it has more, sample a deterministic
source-balanced subset. The allprev source is already obfuscated, so runner
JSON files are generated only for prompts_417 and prompts_335.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

SOURCES = {
    "allprev": BASE / "results" / "allprev_6d_classification" / "allprev_6d_classification.csv",
    "prompts417": BASE / "results" / "prompts_417_6d_classification" / "prompts_417_6d_classification.csv",
    "prompts335": BASE / "results" / "prompts_335_6d_classification" / "prompts_335_6d_classification.csv",
}

CATEGORIES = {
    "task_instruction": "Task Instruction",
    "format_spec": "Format Spec",
    "constraints": "Constraints",
    "context_knowledge": "Context Knowledge",
    "tool_protocol": "Tool/Protocol",
}

DIMENSIONS = [
    "Role_Persona",
    "Task_Instruction",
    "Format_Specification",
    "Constraints_Guardrails",
    "Context_Knowledge",
    "Tool_Action_Protocol",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def read_rows() -> list[dict[str, str]]:
    rows = []
    for source, path in SOURCES.items():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source"] = source
                rows.append(row)
    return rows


def balanced_sample(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return list(rows)

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["_source"]].append(row)

    rng = random.Random(seed)
    for source_rows in by_source.values():
        rng.shuffle(source_rows)

    total = len(rows)
    allocation = {}
    fractions = []
    for source, source_rows in by_source.items():
        raw = limit * len(source_rows) / total
        base = min(len(source_rows), int(raw))
        allocation[source] = base
        fractions.append((raw - base, source))

    remaining = limit - sum(allocation.values())
    for _, source in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if allocation[source] < len(by_source[source]):
            allocation[source] += 1
            remaining -= 1

    selected = []
    for source in sorted(by_source):
        selected.extend(by_source[source][: allocation[source]])
    return selected


def runner_record(row: dict[str, str]) -> dict:
    return {
        "id": row["id"],
        "prompt_label": pick(row, "prompt_name", "prompt_label", "original_prompt_category"),
        "paper_title": pick(row, "title", "paper_title", "paper_file"),
        "prompt_text": row["text"],
        "scores": {dim: float(row.get(dim, 0.0) or 0.0) for dim in DIMENSIONS},
        "source": {
            "source": row["_source"],
            "source_id": row.get("source_id", ""),
            "paper_file": row.get("paper_file", ""),
            "venue": row.get("venue", ""),
            "page": row.get("page", ""),
            "source_group": row.get("source_group", ""),
            "source_batch_file": row.get("source_batch_file", ""),
            "original_prompt_category": row.get("original_prompt_category", ""),
            "text_sha256_16": row.get("text_sha256_16", ""),
        },
    }


def main() -> int:
    args = parse_args()
    rows = read_rows()
    out_root = BASE / "results" / "remaining_5category_selection"
    out_root.mkdir(parents=True, exist_ok=True)
    prompt_dir = BASE / "system prompt"

    summary = {}
    for slug, category in CATEGORIES.items():
        category_rows = [row for row in rows if row["category"] == category]
        selected = balanced_sample(category_rows, args.max_per_category, args.seed)
        selected_ids = [row["id"] for row in selected]

        (out_root / f"{slug}_prompt_ids.json").write_text(
            json.dumps(selected_ids, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        source_counts = defaultdict(int)
        selected_source_counts = defaultdict(int)
        for row in category_rows:
            source_counts[row["_source"]] += 1
        for row in selected:
            selected_source_counts[row["_source"]] += 1

        summary[slug] = {
            "category": category,
            "available_total": len(category_rows),
            "selected_total": len(selected),
            "available_by_source": dict(sorted(source_counts.items())),
            "selected_by_source": dict(sorted(selected_source_counts.items())),
            "ids_path": str((out_root / f"{slug}_prompt_ids.json").relative_to(BASE)),
        }

        for source in ("prompts417", "prompts335"):
            runner = [runner_record(row) for row in selected if row["_source"] == source]
            out_path = prompt_dir / f"{source}_{slug}_runner.json"
            out_path.write_text(json.dumps(runner, indent=2, ensure_ascii=False), encoding="utf-8")
            summary[slug][f"{source}_runner"] = str(out_path.relative_to(BASE))
            summary[slug][f"{source}_runner_n"] = len(runner)

    (out_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
