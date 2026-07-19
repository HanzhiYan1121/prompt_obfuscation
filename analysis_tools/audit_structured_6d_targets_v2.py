#!/usr/bin/env python3
"""Compare regex v1 targets with teacher-generated v2 structured targets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


BASE = Path(__file__).resolve().parents[1]
FIELDS = ["role", "task", "format", "constraints", "context", "tool_protocol"]
CATEGORIES = [
    ("role_persona", "Role/Persona", "role"),
    ("task_instruction", "Task Instruction", "task"),
    ("format_spec", "Format Spec", "format"),
    ("constraints", "Constraints", "constraints"),
    ("context_knowledge", "Context Knowledge", "context"),
    ("tool_protocol", "Tool/Protocol", "tool_protocol"),
]
TOKEN = re.compile(r"[A-Za-z0-9_{}<>\[\]-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_prefix", default="sixcat150")
    parser.add_argument("--samples_per_category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/sixcat150_structured_targets_v2_audit")
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split()).strip(" .;,:\"'")


def tokens(text: str) -> set[str]:
    return {item.lower() for item in TOKEN.findall(text) if len(item) > 2}


def near_duplicate_pairs(target: dict[str, str], threshold: float = 0.8) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(FIELDS):
        a = tokens(target.get(left, ""))
        if len(a) < 4:
            continue
        for right in FIELDS[i + 1 :]:
            b = tokens(target.get(right, ""))
            if len(b) < 4:
                continue
            score = len(a & b) / len(a | b)
            if score >= threshold:
                pairs.append((left, right))
    return pairs


def exact_duplicate_groups(target: dict[str, str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for field in FIELDS:
        value = normalize(target.get(field, ""))
        if value:
            groups.setdefault(value, []).append(field)
    return [group for group in groups.values() if len(group) > 1]


def target_metrics(target: dict[str, str], original: str, dominant: str) -> dict[str, Any]:
    values = [target.get(field, "") for field in FIELDS]
    total_chars = sum(len(value) for value in values)
    original_norm = normalize(original)
    copied_full = sum(normalize(value) == original_norm for value in values if normalize(value))
    return {
        "nonempty_fields": sum(bool(value.strip()) for value in values),
        "dominant_nonempty": bool(target.get(dominant, "").strip()),
        "exact_duplicate_groups": exact_duplicate_groups(target),
        "near_duplicate_pairs": near_duplicate_pairs(target),
        "full_prompt_copy_fields": copied_full,
        "target_chars": total_chars,
        "compression_ratio": total_chars / max(1, len(original)),
    }


def summarize(rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    metrics = [row[version] for row in rows]
    return {
        "records": len(metrics),
        "mean_nonempty_fields": round(mean(m["nonempty_fields"] for m in metrics), 3),
        "dominant_field_coverage": round(mean(m["dominant_nonempty"] for m in metrics), 3),
        "records_with_exact_duplicates": sum(bool(m["exact_duplicate_groups"]) for m in metrics),
        "records_with_near_duplicates": sum(bool(m["near_duplicate_pairs"]) for m in metrics),
        "records_copying_full_prompt": sum(bool(m["full_prompt_copy_fields"]) for m in metrics),
        "mean_compression_ratio": round(mean(m["compression_ratio"] for m in metrics), 3),
    }


def short_json(target: dict[str, str]) -> str:
    return json.dumps(target, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else BASE / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    all_rows: list[dict[str, Any]] = []
    category_summary: dict[str, Any] = {}
    audit_sections: list[str] = [
        "# Structured Target v1 vs v2 Audit",
        "",
        "V1 uses overlapping regex matches. V2 uses evidence-grounded semantic decomposition by a teacher model. V2 remains silver annotation and requires manual review.",
        "",
    ]

    for slug, label, dominant in CATEGORIES:
        v1_path = BASE / f"results/{args.result_prefix}_{slug}_structured_recovery/recovery_inputs.json"
        v2_path = BASE / f"results/{args.result_prefix}_{slug}_structured_recovery_v2/recovery_inputs.json"
        v1_records = json.loads(v1_path.read_text(encoding="utf-8"))
        v2_records = json.loads(v2_path.read_text(encoding="utf-8"))
        v1_by_id = {str(record["id"]): record for record in v1_records}
        rows: list[dict[str, Any]] = []
        for v2 in v2_records:
            record_id = str(v2["id"])
            v1 = v1_by_id[record_id]
            original = v2["target_original_prompt"]
            old_target = v1["target_structured_6d"]
            new_target = v2["target_structured_6d_v2"]
            row = {
                "id": record_id,
                "category": label,
                "dominant_field": dominant,
                "original_prompt": original,
                "v1_target": old_target,
                "v2_target": new_target,
                "v1": target_metrics(old_target, original, dominant),
                "v2": target_metrics(new_target, original, dominant),
                "v2_validation": v2.get("target_structured_6d_v2_validation", {}),
            }
            rows.append(row)
            all_rows.append(row)

        category_summary[slug] = {
            "category": label,
            "v1": summarize(rows, "v1"),
            "v2": summarize(rows, "v2"),
        }

        # Prioritize records where v1 duplicated content, then add a seeded sample.
        priority = [
            row
            for row in rows
            if row["v1"]["exact_duplicate_groups"] or row["v1"]["near_duplicate_pairs"]
        ]
        rng.shuffle(priority)
        remaining = [row for row in rows if row not in priority]
        rng.shuffle(remaining)
        selected = (priority + remaining)[: args.samples_per_category]
        audit_sections.extend([f"## {label}", ""])
        for row in selected:
            audit_sections.extend(
                [
                    f"### {row['id']}",
                    "",
                    "**Original prompt**",
                    "",
                    "```text",
                    row["original_prompt"][:4000],
                    "```",
                    "",
                    "**V1 target**",
                    "",
                    "```json",
                    short_json(row["v1_target"]),
                    "```",
                    "",
                    "**V2 target**",
                    "",
                    "```json",
                    short_json(row["v2_target"]),
                    "```",
                    "",
                    "Manual verdict: `correct / minor edit / major error`",
                    "",
                ]
            )

    overall = {
        "v1": summarize(all_rows, "v1"),
        "v2": summarize(all_rows, "v2"),
    }
    report = {"overall": overall, "by_category": category_summary}
    (output_dir / "target_quality_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "manual_audit.md").write_text("\n".join(audit_sections), encoding="utf-8")

    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"Wrote audit outputs to {output_dir.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
