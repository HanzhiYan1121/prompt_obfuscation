#!/usr/bin/env python3
"""Prepare a category-specific runner JSON from a 6D classification CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--classification_csv", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def main() -> int:
    args = parse_args()
    csv_path = resolve(args.classification_csv)
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] != args.category:
                continue
            prompts.append(
                {
                    "id": row["id"],
                    "prompt_label": row["prompt_name"] or row["original_prompt_category"],
                    "paper_title": row["title"] or row["paper_file"],
                    "prompt_text": row["text"],
                    "scores": {dim: float(row[dim]) for dim in DIMENSIONS},
                    "source": {
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
            )

    out_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(prompts)} {args.category} prompts to {out_path.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
