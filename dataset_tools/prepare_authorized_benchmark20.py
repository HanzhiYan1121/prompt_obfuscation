#!/usr/bin/env python3
"""Select a deterministic, recovery-held-out 20-prompt benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CATEGORY_QUOTAS = {
    "role_persona": 4,
    "task_instruction": 4,
    "format_spec": 3,
    "constraints": 3,
    "context_knowledge": 3,
    "tool_protocol": 3,
}
MODEL_TAG = "qwen3_32b"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def choose_records(seed: int) -> tuple[list[dict], list[dict], dict]:
    selected_manifest: list[dict] = []
    runner: list[dict] = []
    selected_ids: set[str] = set()
    audit: dict[str, dict] = {}

    for category_index, (category, quota) in enumerate(CATEGORY_QUOTAS.items()):
        lora_dir = BASE / "results" / f"sixcat200_{category}_structured_{MODEL_TAG}_lora"
        inputs_path = BASE / "results" / f"sixcat200_{category}_structured_recovery" / "recovery_inputs.json"
        split_path = lora_dir / "split.json"
        adapter_path = lora_dir / "adapter" / "adapter_model.safetensors"
        for required in (inputs_path, split_path, adapter_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(split["train_ids"])
        test_ids = list(split["test_ids"])
        if train_ids & set(test_ids):
            raise ValueError(f"Train/test overlap in {split_path}")
        by_id = {
            row["id"]: row
            for row in json.loads(inputs_path.read_text(encoding="utf-8"))
        }
        random.Random(seed + category_index).shuffle(test_ids)

        chosen: list[str] = []
        for prompt_id in test_ids:
            if prompt_id in selected_ids or prompt_id in train_ids:
                continue
            row = by_id.get(prompt_id)
            if row is None:
                continue
            prompt_text = str(row.get("target_original_prompt") or "").strip()
            if not prompt_text:
                continue
            source_result_dir = resolve(Path(row["result_dir"]))
            prepared_test = source_result_dir / "prepared_data" / "test_data.json"
            best_candidate = source_result_dir / "best_candidate.pt"
            if not prepared_test.is_file() or not best_candidate.is_file():
                continue
            queries = json.loads(prepared_test.read_text(encoding="utf-8"))
            if len(queries) < 16:
                continue

            chosen.append(prompt_id)
            selected_ids.add(prompt_id)
            selected_manifest.append(
                {
                    "id": prompt_id,
                    "category": category,
                    "recovery_train_disjoint": True,
                    "source_result_dir": str(source_result_dir.relative_to(BASE)),
                    "source_recovery_inputs": str(inputs_path.relative_to(BASE)),
                    "recovery_lora_dir": str(lora_dir.relative_to(BASE)),
                    "split_path": str(split_path.relative_to(BASE)),
                    "prompt_sha256": sha256_text(prompt_text),
                    "prompt_char_len": len(prompt_text),
                    "available_query_count": len(queries),
                }
            )
            runner.append(
                {
                    "id": prompt_id,
                    "prompt_text": prompt_text,
                    "prompt_label": row.get("prompt_label"),
                    "paper_title": row.get("paper_title"),
                    "scores": row.get("scores", {}),
                    "benchmark_category": category,
                }
            )
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f"Could select only {len(chosen)}/{quota} for {category}")
        audit[category] = {
            "quota": quota,
            "selected_ids": chosen,
            "adapter": str(adapter_path.relative_to(BASE)),
            "train_n": len(train_ids),
            "test_n": len(test_ids),
            "selected_train_overlap": len(set(chosen) & train_ids),
        }

    if len(selected_manifest) != 20 or len(selected_ids) != 20:
        raise ValueError("Selection is not exactly 20 unique prompts")
    return selected_manifest, runner, audit


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    selected, runner, audit = choose_records(args.seed)
    report = {
        "seed": args.seed,
        "total": len(selected),
        "category_quotas": CATEGORY_QUOTAS,
        "query_partition": {
            "recovery_evidence_indices": [0, 5],
            "functional_evaluation_indices": [6, 15],
        },
        "records": selected,
        "audit": audit,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        if output_dir.exists():
            raise FileExistsError(f"Dry-run output target already exists: {output_dir}")
        return 0
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "runner.json").write_text(
        json.dumps(runner, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "selected_ids.json").write_text(
        json.dumps([row["id"] for row in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ids_dir = output_dir / "ids_by_category"
    ids_dir.mkdir()
    for category in CATEGORY_QUOTAS:
        category_ids = [row["id"] for row in selected if row["category"] == category]
        (ids_dir / f"{category}.json").write_text(
            json.dumps(category_ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
