#!/usr/bin/env python3
"""Apply deterministic conservative refinement to generated v2 silver targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_structured_6d_recovery_targets_v2 import (
    FIELDS,
    finalize_target,
    validate_target,
    write_report,
)


BASE = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "role_persona": "role",
    "task_instruction": "task",
    "format_spec": "format",
    "constraints": "constraints",
    "context_knowledge": "context",
    "tool_protocol": "tool_protocol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_prefix", default="sixcat150")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for slug, dominant in CATEGORIES.items():
        path = BASE / f"results/{args.result_prefix}_{slug}_structured_recovery_v2/recovery_inputs.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            prompt = record["target_original_prompt"]
            target = record["target_structured_6d_v2"]
            target = finalize_target(target, prompt, dominant)
            errors, metrics = validate_target(target, prompt, dominant)
            record["target_structured_6d_v2"] = target
            record["target_structured_6d_v2_json"] = json.dumps(
                target, ensure_ascii=False, sort_keys=True
            )
            previous = record.get("target_structured_6d_v2_validation", {})
            record["target_structured_6d_v2_validation"] = {
                **metrics,
                "errors": errors,
                "retries": previous.get("retries", 0),
                "conservative_refinement": True,
            }
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        with (path.parent / "recovery_inputs.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_report(records, path)
        print(f"Refined {len(records)} records: {path.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
