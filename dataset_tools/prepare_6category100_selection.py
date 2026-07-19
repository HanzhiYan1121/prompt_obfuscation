#!/usr/bin/env python3
"""Prepare six 6D category prompt sets with a fixed number of prompts each.

Selection policy:
1. Prefer prompts whose dominant 6D category matches the target category.
2. If fewer than target_n prompts are available, fill from the previous prompt
   library using the corresponding dimension score.
3. If more than target_n prompts are available, take a deterministic,
   source-balanced sample.

The script also writes runner JSON files for prompts that still need
obfuscation. Existing obfuscation outputs are reused by later recovery steps.
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
    "role_persona": ("Role/Persona", "Role_Persona"),
    "task_instruction": ("Task Instruction", "Task_Instruction"),
    "format_spec": ("Format Spec", "Format_Specification"),
    "constraints": ("Constraints", "Constraints_Guardrails"),
    "context_knowledge": ("Context Knowledge", "Context_Knowledge"),
    "tool_protocol": ("Tool/Protocol", "Tool_Action_Protocol"),
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
    parser.add_argument("--target_n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output selection directory. Defaults to results/six_category_<target_n>_selection.",
    )
    parser.add_argument(
        "--runner_prefix",
        default=None,
        help="Prefix for generated runner files under system prompt/. Defaults to 6cat<target_n>.",
    )
    return parser.parse_args()


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source, path in SOURCES.items():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source"] = source
                rows.append(row)
    return rows


def pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def score(row: dict[str, str], dimension: str) -> float:
    try:
        return float(row.get(dimension, 0.0) or 0.0)
    except ValueError:
        return 0.0


def balanced_downsample(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return list(rows)

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["_source"]].append(row)

    rng = random.Random(seed)
    for source_rows in by_source.values():
        rng.shuffle(source_rows)

    total = len(rows)
    allocation: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
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

    selected: list[dict[str, str]] = []
    for source in sorted(by_source):
        selected.extend(by_source[source][: allocation[source]])
    return selected


def existing_obfuscation_ids() -> set[str]:
    ids: set[str] = set()
    for meta_path in (BASE / "precomputed_results").glob("*_tok10/prompt_metadata.json"):
        if not (meta_path.parent / "best_candidate.pt").exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt_id = metadata.get("id")
        if prompt_id:
            ids.add(prompt_id)
    return ids


def existing_prompt_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for meta_path in (BASE / "precomputed_results").glob("*_tok10/prompt_metadata.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt_id = metadata.get("id")
        prompt_path = meta_path.parent / "original_prompt.txt"
        if prompt_id and prompt_path.exists() and prompt_id not in texts:
            texts[prompt_id] = prompt_path.read_text(encoding="utf-8")
    return texts


PROMPT_TEXTS = existing_prompt_texts()


def prompt_text(row: dict[str, str]) -> str:
    return row.get("text") or PROMPT_TEXTS.get(row["id"], "")


def runner_record(row: dict[str, str]) -> dict:
    return {
        "id": row["id"],
        "prompt_label": pick(row, "prompt_name", "prompt_label", "original_prompt_category"),
        "paper_title": pick(row, "title", "paper_title", "paper_file"),
        "prompt_text": prompt_text(row),
        "scores": {dim: score(row, dim) for dim in DIMENSIONS},
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
            "selected_category": row.get("_selected_category", ""),
            "selected_dimension": row.get("_selected_dimension", ""),
            "selection_reason": row.get("_selection_reason", ""),
        },
    }


def select_category(
    all_rows: list[dict[str, str]],
    category: str,
    dimension: str,
    target_n: int,
    seed: int,
) -> list[dict[str, str]]:
    primary = [row for row in all_rows if row["category"] == category]
    primary = balanced_downsample(primary, target_n, seed)

    selected_by_id = {row["id"]: dict(row, _selection_reason="dominant_category") for row in primary}
    if len(selected_by_id) >= target_n:
        return list(selected_by_id.values())[:target_n]

    candidates = [
        row
        for row in all_rows
        if row["id"] not in selected_by_id and score(row, dimension) > 0
    ]
    candidates.sort(
        key=lambda row: (
            score(row, dimension),
            row["category"] == "Mixed/Complex",
            row["_source"] == "allprev",
            row["id"],
        ),
        reverse=True,
    )
    for row in candidates:
        selected_by_id[row["id"]] = dict(row, _selection_reason="dimension_score_fill")
        if len(selected_by_id) >= target_n:
            break
    return list(selected_by_id.values())[:target_n]


def main() -> int:
    args = parse_args()
    rows = read_rows()
    existing_ids = existing_obfuscation_ids()
    out_root = args.output_dir if args.output_dir is not None else BASE / "results" / f"six_category_{args.target_n}_selection"
    if not out_root.is_absolute():
        out_root = BASE / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    prompt_dir = BASE / "system prompt"
    runner_prefix = args.runner_prefix or f"6cat{args.target_n}"

    summary = {}
    for slug, (category, dimension) in CATEGORIES.items():
        selected = select_category(rows, category, dimension, args.target_n, args.seed)
        for row in selected:
            row["_selected_category"] = category
            row["_selected_dimension"] = dimension

        ids = [row["id"] for row in selected]
        (out_root / f"{slug}_prompt_ids.json").write_text(
            json.dumps(ids, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_root / f"{slug}_selected_prompts.json").write_text(
            json.dumps([runner_record(row) for row in selected], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        source_counts = defaultdict(int)
        category_counts = defaultdict(int)
        reason_counts = defaultdict(int)
        missing_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in selected:
            source_counts[row["_source"]] += 1
            category_counts[row["category"]] += 1
            reason_counts[row.get("_selection_reason", "")] += 1
            if row["_source"] in {"prompts417", "prompts335"} and row["id"] not in existing_ids:
                missing_by_source[row["_source"]].append(row)

        summary[slug] = {
            "category": category,
            "dimension": dimension,
            "selected_total": len(selected),
            "selected_by_source": dict(sorted(source_counts.items())),
            "selected_by_original_category": dict(sorted(category_counts.items())),
            "selection_reasons": dict(sorted(reason_counts.items())),
            "ids_path": str((out_root / f"{slug}_prompt_ids.json").relative_to(BASE)),
            "selected_prompts_path": str((out_root / f"{slug}_selected_prompts.json").relative_to(BASE)),
        }

        for source in ("prompts417", "prompts335"):
            runner = [runner_record(row) for row in missing_by_source[source]]
            out_path = prompt_dir / f"{source}_{runner_prefix}_{slug}_runner.json"
            out_path.write_text(json.dumps(runner, indent=2, ensure_ascii=False), encoding="utf-8")
            summary[slug][f"{source}_missing_obfuscation_n"] = len(runner)
            summary[slug][f"{source}_missing_runner"] = str(out_path.relative_to(BASE))

    (out_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
