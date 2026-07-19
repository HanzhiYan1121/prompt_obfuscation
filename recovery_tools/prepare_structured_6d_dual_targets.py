#!/usr/bin/env python3
"""Add a directly executable prompt to the cleaned six-field v2 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
FIELDS = ["role", "task", "format", "constraints", "context", "tool_protocol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_prefix", default="sixcat150")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for slug in [
        "role_persona",
        "task_instruction",
        "format_spec",
        "constraints",
        "context_knowledge",
        "tool_protocol",
    ]:
        source = BASE / f"results/{args.result_prefix}_{slug}_structured_recovery_v2/recovery_inputs.json"
        output = BASE / f"results/{args.result_prefix}_{slug}_structured_recovery_dual/recovery_inputs.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        for record in records:
            structured = record["target_structured_6d_v2"]
            dual = {field: structured.get(field, "") for field in FIELDS}
            dual["executable_prompt"] = record["target_original_prompt"].strip()
            record["target_structured_6d_dual"] = dual
            record["target_structured_6d_dual_json"] = json.dumps(
                dual, ensure_ascii=False, sort_keys=True
            )
            record["target_structured_6d_dual_meta"] = {
                "structured_source": "target_structured_6d_v2",
                "executable_source": "target_original_prompt",
                "purpose": "joint_structure_and_function_supervision",
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        with (output.parent / "recovery_inputs.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} dual targets: {output.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
