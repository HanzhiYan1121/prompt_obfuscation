#!/usr/bin/env python3
"""Convert all_previous_system_prompts_combined.json to runner input format."""

from __future__ import annotations

import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "system prompt" / "all_previous_system_prompts_combined.json"
OUT = BASE / "system prompt" / "all_previous_system_prompts_runner.json"


def short(text: str, limit: int = 36) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (text[:limit].strip("_") or "prompt")


def main() -> int:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("prompts", []):
        prompt_text = (item.get("text") or "").strip()
        if not prompt_text:
            continue
        gid = int(item["global_id"])
        label = item.get("label") or f"prompt_{gid:03d}"
        paper_title = item.get("paper_title") or ""
        rows.append(
            {
                "id": f"AllPrev-{gid:03d} ({short(label, 28)})",
                "prompt_label": label,
                "paper_title": paper_title,
                "prompt_text": prompt_text,
                "source_file": item.get("file"),
                "source_group": item.get("source_group"),
                "venue": item.get("venue"),
                "text_sha256_16": item.get("text_sha256_16"),
                "scores": {},
            }
        )
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} prompts -> {OUT.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
