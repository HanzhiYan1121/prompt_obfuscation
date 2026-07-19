#!/usr/bin/env python3
"""Prepare Role/Persona prompts from prompts_417 for the obfuscation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CLASSIFICATION = BASE / "results" / "prompts_417_6d_classification" / "prompts_417_6d_classification.csv"
OUT = BASE / "system prompt" / "prompts_417_role_runner.json"

DIMENSIONS = [
    "Role_Persona",
    "Task_Instruction",
    "Format_Specification",
    "Constraints_Guardrails",
    "Context_Knowledge",
    "Tool_Action_Protocol",
]


def main() -> int:
    prompts = []
    with CLASSIFICATION.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] != "Role/Persona":
                continue
            prompts.append(
                {
                    "id": row["id"],
                    "prompt_label": row["prompt_name"] or row["original_prompt_category"],
                    "paper_title": row["title"] or row["paper_file"],
                    "prompt_text": row["text"],
                    "scores": {dim: float(row[dim]) for dim in DIMENSIONS},
                    "source": {
                        "paper_file": row["paper_file"],
                        "venue": row["venue"],
                        "source_group": row["source_group"],
                        "source_prompt_file": row["source_prompt_file"],
                        "original_prompt_category": row["original_prompt_category"],
                        "text_sha256_16": row["text_sha256_16"],
                    },
                }
            )

    OUT.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(prompts)} Role/Persona prompts to {OUT.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
