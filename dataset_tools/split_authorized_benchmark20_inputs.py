#!/usr/bin/env python3
"""Split one 20-prompt recovery-input file by its assigned recovery adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(resolve(args.manifest).read_text(encoding="utf-8"))
    records = json.loads(resolve(args.input).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate IDs in recovery inputs")
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)

    categories = manifest["category_quotas"]
    total = 0
    for category in categories:
        ids = [row["id"] for row in manifest["records"] if row["category"] == category]
        missing = [prompt_id for prompt_id in ids if prompt_id not in by_id]
        if missing:
            raise KeyError(f"Missing {category} recovery inputs: {missing}")
        subset = [by_id[prompt_id] for prompt_id in ids]
        total += len(subset)
        (output_dir / f"{category}.json").write_text(
            json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if total != 20:
        raise ValueError(f"Expected 20 split records, got {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
