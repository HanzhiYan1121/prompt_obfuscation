#!/usr/bin/env python3
"""Add six-dimensional structured JSON recovery targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def first_match(lines: list[str], patterns: list[str], max_chars: int = 520) -> str:
    hits = []
    for line in lines:
        if any(re.search(pattern, line, flags=re.I) for pattern in patterns):
            hits.append(line)
    return " ".join(hits)[:max_chars].strip()


def extract_structure(prompt: str) -> dict[str, str]:
    lines = sentences(prompt)
    role = first_match(
        lines,
        [
            r"\byou are\b",
            r"\bact as\b",
            r"\byour role\b",
            r"\brole:",
            r"\bexpert\b",
            r"\bassistant\b",
            r"\bevaluator\b",
            r"\bjudge\b",
        ],
        420,
    )
    task = first_match(
        lines,
        [
            r"\byour task\b",
            r"\btask:",
            r"\bplease\b",
            r"\bgenerate\b",
            r"\bevaluate\b",
            r"\bclassify\b",
            r"\banswer\b",
            r"\bselect\b",
            r"\bextract\b",
            r"\bwrite\b",
            r"\bsummarize\b",
            r"\brank\b",
            r"\bsolve\b",
        ],
        560,
    )
    fmt = first_match(
        lines,
        [
            r"\boutput\b",
            r"\bformat\b",
            r"\bjson\b",
            r"\breturn\b",
            r"\bschema\b",
            r"\banswer:",
            r"\bscore:",
            r"\btable\b",
            r"\blist\b",
            r"\bfield\b",
        ],
        520,
    )
    constraints = first_match(
        lines,
        [
            r"\bmust\b",
            r"\bonly\b",
            r"\bdo not\b",
            r"\bdon't\b",
            r"\bnever\b",
            r"\bwithout\b",
            r"\bstrict\b",
            r"\bavoid\b",
            r"\brequired\b",
            r"\bno more than\b",
        ],
        560,
    )
    context = first_match(
        lines,
        [
            r"\bcontext\b",
            r"\bgiven\b",
            r"\bbackground\b",
            r"\bpassage\b",
            r"\bdocument\b",
            r"\barticle\b",
            r"\bclaim\b",
            r"\bquestion\b",
            r"\bdialogue\b",
            r"\bhistory\b",
            r"\bexamples?\b",
        ],
        620,
    )
    tool_protocol = first_match(
        lines,
        [
            r"\btool\b",
            r"\bapi\b",
            r"\bfunction\b",
            r"\bcall\b",
            r"\bpython\b",
            r"\bcode\b",
            r"\bworkflow\b",
            r"\bprotocol\b",
            r"\bstep\b",
            r"\bquery\b",
            r"\bsearch\b",
        ],
        560,
    )
    if not task and lines:
        task = lines[0][:320]
    return {
        "role": role,
        "task": task,
        "format": fmt,
        "constraints": constraints,
        "context": context,
        "tool_protocol": tool_protocol,
    }


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    for record in records:
        target = extract_structure(record["target_original_prompt"])
        record["target_structured_6d"] = target
        record["target_structured_6d_json"] = json.dumps(target, ensure_ascii=False, sort_keys=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_path.parent / "recovery_inputs.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} structured 6D recovery records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
