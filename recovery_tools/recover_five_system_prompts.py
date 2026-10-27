#!/usr/bin/env python3
"""Shared prompt construction for the local prompt-recovery experiments.

The recovery scripts import this module directly because they are executed from
``recovery_tools/``.  It intentionally formats only evidence already present in
the local recovery-input records.
"""

from __future__ import annotations

import json


SYSTEM_MESSAGE = (
    "You are a system-prompt recovery analyst. Given local evidence from a "
    "protected prompt, reconstruct the most likely system-prompt instructions. "
    "Use only the supplied evidence and do not add commentary."
)


def _score_text(scores: object) -> str:
    if not isinstance(scores, dict):
        return "{}"
    return json.dumps(scores, ensure_ascii=False, sort_keys=True)


def build_recovery_prompt(
    record: dict,
    mode: str = "metadata_behavior_projection",
    n_examples: int = 6,
) -> str:
    """Build the deterministic evidence template used by recovery inference."""
    if mode != "metadata_behavior_projection":
        raise ValueError(f"Unsupported recovery mode: {mode}")

    obfuscation = record.get("obfuscation") or {}
    examples = list(record.get("behavior_examples") or [])[:n_examples]
    lines = [
        "Recover the hidden system prompt from the following local evidence.",
        "Focus on behavior that is consistently caused by the protected prompt, especially differences from the ordinary model output.",
        "",
        "Metadata:",
        f"- Prompt label: {record.get('prompt_label') or ''}",
        f"- Paper title: {record.get('paper_title') or ''}",
        f"- Six-dimension scores: {_score_text(record.get('scores'))}",
        "",
        "Nearest-token projections of the protected soft prompt:",
        f"- Euclidean: {obfuscation.get('projection_euclidean_text') or ''}",
        f"- Cosine: {obfuscation.get('projection_cosine_text') or ''}",
        "",
        "Observed behavior examples:",
    ]
    if not examples:
        lines.append("- No behavior examples available.")
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"Example {index}",
                f"User input: {example.get('user_input') or ''}",
                f"Protected output: {example.get('obfuscated_output') or ''}",
            ]
        )
        if example.get("conventional_output") is not None:
            lines.append(f"Ordinary output without the protected prompt: {example.get('conventional_output') or ''}")
    lines.extend(["", "Return the recovered prompt only."])
    return "\n".join(lines)
