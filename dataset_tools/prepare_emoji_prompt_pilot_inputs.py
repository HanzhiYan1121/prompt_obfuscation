#!/usr/bin/env python3
"""Build a small EmojiPrompt-style recovery pilot dataset.

This script reuses the existing six-category prompt selections and matched
test queries. It creates a symbolic / emoji-style obfuscated system prompt,
runs the base model under that obfuscated prompt, and writes recovery inputs
compatible with the existing structured QLoRA pipeline.

This is a lightweight pilot, not a full reproduction of the EmojiPrompt paper:
there is no LDP post-sampling and no learned obfuscation prompt optimization.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from torch.utils.data import DataLoader

from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses


BASE = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "role_persona",
    "task_instruction",
    "format_spec",
    "constraints",
    "context_knowledge",
    "tool_protocol",
]

WORD_SYMBOLS = {
    "you": "👤",
    "are": "≈",
    "assistant": "🤖",
    "agent": "🧭",
    "expert": "🎓",
    "doctor": "🩺",
    "medical": "🩺",
    "lawyer": "⚖️",
    "teacher": "📚",
    "student": "🎒",
    "user": "👤",
    "answer": "💬",
    "question": "❓",
    "task": "🎯",
    "classify": "🏷️",
    "classification": "🏷️",
    "extract": "🔎",
    "summarize": "📝",
    "summary": "📝",
    "translate": "🌐",
    "generate": "⚙️",
    "write": "✍️",
    "respond": "💬",
    "format": "📐",
    "json": "{}",
    "list": "☷",
    "table": "▦",
    "output": "📤",
    "return": "↩",
    "only": "⊂",
    "do": "▶",
    "not": "¬",
    "never": "🚫",
    "avoid": "🚫",
    "must": "!",
    "should": "~",
    "safe": "🛡️",
    "private": "🔒",
    "confidential": "🔒",
    "context": "📎",
    "evidence": "📌",
    "document": "📄",
    "passage": "📄",
    "tool": "🛠️",
    "function": "ƒ",
    "api": "🔌",
    "call": "📞",
    "step": "①",
    "protocol": "📜",
    "reason": "∵",
    "label": "🏷️",
}

CATEGORY_HINTS = {
    "role_persona": "🎭 role/persona signal",
    "task_instruction": "🎯 task/action signal",
    "format_spec": "📐 output-format signal",
    "constraints": "🛡️ rule/guardrail signal",
    "context_knowledge": "📎 context/background signal",
    "tool_protocol": "🛠️ tool/protocol signal",
}

TOKEN_RE = re.compile(r"\{[^}]+\}|[A-Za-z0-9_./:-]+|[^\sA-Za-z0-9_./:-]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_dir", type=Path, default=Path("results/six_category_200_selection"))
    parser.add_argument("--sample_n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--output_prefix", default="emoji_pilot50")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--categories", default=" ".join(CATEGORIES))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def precomputed_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for meta_path in (BASE / "precomputed_results").glob("*_tok10/prompt_metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt_id = meta.get("id")
        if (
            prompt_id
            and (meta_path.parent / "prepared_data" / "test_data.json").exists()
            and (meta_path.parent / "conventional_output.json").exists()
        ):
            out.setdefault(prompt_id, meta_path.parent)
    return out


def compact_symbolic_prompt(prompt: str, category: str) -> str:
    """Convert text into a compact symbolic sketch while preserving placeholders."""
    pieces: list[str] = []
    for tok in TOKEN_RE.findall(prompt):
        if tok.startswith("{") and tok.endswith("}"):
            pieces.append(tok)
            continue
        key = tok.lower().strip(".,;:!?()[]\"'")
        if key in WORD_SYMBOLS:
            pieces.append(WORD_SYMBOLS[key])
        elif len(key) <= 2:
            pieces.append(tok)
        elif key.isdigit():
            pieces.append("#")
        elif re.fullmatch(r"[A-Za-z0-9_./:-]+", tok):
            # Keep a weak abbreviation rather than the full lexical token.
            pieces.append(key[:2] + "·")
        else:
            pieces.append(tok)
    symbolic = " ".join(pieces)
    symbolic = re.sub(r"\s+", " ", symbolic).strip()
    return f"{CATEGORY_HINTS.get(category, '🔐 symbolic prompt')}\n{symbolic}"


def semantic_symbol_sketch(prompt: str, category: str) -> str:
    lower = prompt.lower()
    hits = [word for word in WORD_SYMBOLS if word in lower]
    hit_symbols = []
    seen = set()
    for word in hits:
        sym = WORD_SYMBOLS[word]
        if sym not in seen:
            hit_symbols.append(sym)
            seen.add(sym)
    placeholders = re.findall(r"\{[^}]+\}", prompt)
    return " ".join([CATEGORY_HINTS.get(category, "🔐"), *hit_symbols[:24], *placeholders[:8]]).strip()


def load_queries_and_refs(result_dir: Path, limit: int) -> tuple[list[str], list[str]]:
    queries = json.loads((result_dir / "prepared_data" / "test_data.json").read_text(encoding="utf-8"))[:limit]
    conv = json.loads((result_dir / "conventional_output.json").read_text(encoding="utf-8"))
    refs = []
    for group in conv.get("output", [])[:limit]:
        refs.append(str(group[0]) if group else "")
    return [str(q) for q in queries], refs


def generate_outputs(model: Model, system_prompt: str, queries: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    dataset = TextDataset(queries)
    collate_fn = create_collate_fn(tokenizer=model.tokenizer, system_prompt=system_prompt)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    generation_args = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 100,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
    }
    raw = generate_model_responses(model, dataloader, generation_args)
    return [str(group[0]) if group else "" for group in raw]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    selection_dir = resolve(args.selection_dir)
    categories = [c for c in args.categories.split() if c]
    index = precomputed_index()

    # Choose a model from the first usable prompt unless the user supplied one.
    first_model_name = args.model_name
    if not first_model_name:
        for slug in categories:
            rows = json.loads((selection_dir / f"{slug}_selected_prompts.json").read_text(encoding="utf-8"))
            for row in rows:
                result_dir = index.get(row["id"])
                if result_dir and (result_dir / "params.json").exists():
                    first_model_name = json.loads((result_dir / "params.json").read_text(encoding="utf-8"))["model_name"]
                    break
            if first_model_name:
                break
    if not first_model_name:
        raise RuntimeError("Could not infer base model name from precomputed results.")

    model = Model(first_model_name, "4bit")
    summary = {}
    for slug in categories:
        rows = json.loads((selection_dir / f"{slug}_selected_prompts.json").read_text(encoding="utf-8"))
        usable = [row for row in rows if row["id"] in index]
        rng.shuffle(usable)
        selected = sorted(usable[: args.sample_n], key=lambda r: r["id"])
        out_dir = BASE / "results" / f"{args.output_prefix}_{slug}_recovery"
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        print(f"=== [{slug}] selected {len(selected)} prompts ===", flush=True)
        for row in selected:
            result_dir = index[row["id"]]
            original = row.get("prompt_text") or (result_dir / "original_prompt.txt").read_text(encoding="utf-8")
            emoji_prompt = compact_symbolic_prompt(original, slug)
            sketch = semantic_symbol_sketch(original, slug)
            queries, refs = load_queries_and_refs(result_dir, args.n_examples)
            print(f"[{slug}] generating emoji behavior: {row['id']}", flush=True)
            emoji_outputs = generate_outputs(model, emoji_prompt, queries, args.batch_size, args.max_new_tokens)
            examples = [
                {
                    "user_input": q,
                    "obfuscated_output": obf,
                    "conventional_output": ref,
                }
                for q, obf, ref in zip(queries, emoji_outputs, refs)
            ]
            records.append(
                {
                    "id": row["id"],
                    "result_dir": str(result_dir.relative_to(BASE)),
                    "prompt_label": row.get("prompt_label", ""),
                    "paper_title": row.get("paper_title", ""),
                    "scores": row.get("scores", {}),
                    "obfuscation": {
                        "method": "emoji_prompt_style_pilot",
                        "emoji_prompt_text": emoji_prompt,
                        "semantic_symbol_sketch": sketch,
                        # These keys keep compatibility with the old recovery prompt builder.
                        "projection_euclidean_text": emoji_prompt,
                        "projection_cosine_text": sketch,
                    },
                    "behavior_examples": examples,
                    "behavior_source": {
                        "output_tag": "emoji_prompt_style",
                        "probe_category": slug,
                        "probe_set": "matched_original_test",
                    },
                    "target_original_prompt": original,
                    "target_char_len": len(original),
                }
            )
        (out_dir / "recovery_inputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        with (out_dir / "recovery_inputs.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary[slug] = {"records": len(records), "output": str(out_dir.relative_to(BASE))}
    summary_path = BASE / "results" / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
