#!/usr/bin/env python3
"""Collect all Role/Persona prompt ids from 101, 417, and 335 prompt sets."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SOURCES = [
    BASE / "results" / "allprev_6d_classification" / "allprev_6d_classification.csv",
    BASE / "results" / "prompts_417_6d_classification" / "prompts_417_6d_classification.csv",
    BASE / "results" / "prompts_335_6d_classification" / "prompts_335_6d_classification.csv",
]
OUT = BASE / "results" / "allrole_99_recovery" / "role_prompt_ids.json"


def main() -> int:
    ids = []
    counts = {}
    for path in SOURCES:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        role_ids = [row["id"] for row in rows if row["category"] == "Role/Persona"]
        counts[str(path.relative_to(BASE))] = len(role_ids)
        ids.extend(role_ids)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ids, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": counts, "total": len(ids), "out": str(OUT.relative_to(BASE))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
