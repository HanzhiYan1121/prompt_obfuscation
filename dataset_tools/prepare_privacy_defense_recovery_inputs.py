#!/usr/bin/env python3
"""Build recovery inputs for prompt-privacy defense benchmarks.

The goal is to attack several prompt/input protection mechanisms with the same
recovery pipeline used for soft-prompt obfuscation. Some defenses are system
protocols or embedding/split-inference methods, so this script creates an
attackable approximation of the evidence visible to an honest-but-curious cloud:
protected text/sketch plus behavior traces.

This is a benchmark harness, not a full reproduction of every defense paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "role_persona",
    "task_instruction",
    "format_spec",
    "constraints",
    "context_knowledge",
    "tool_protocol",
]

METHODS = [
    "confusionprompt",
    "inferdpt",
    "split_denoise",
    "confidential_prompting",
    "noir",
    "emojicrypt",
    "dp_opt",
    "preempt",
]

METHOD_LABELS = {
    "confusionprompt": "ConfusionPrompt-style mixed real and pseudo sub-prompts",
    "inferdpt": "InferDPT-style token-level perturbed prompt",
    "split_denoise": "Split-and-Denoise-style noisy embedding projection",
    "confidential_prompting": "Confidential Prompting-style virtual prompt set",
    "noir": "NOIR-style randomized tokenizer / IND vocabulary sketch",
    "emojicrypt": "EmojiCrypt-style emoji / symbol encrypted prompt",
    "dp_opt": "DP-OPT-style differentially private prompt sketch",
    "preempt": "PrEEmpt-style sanitized prompt with typed placeholders",
}

SENSITIVE_WORDS = {
    "doctor": "clinician",
    "medical": "health",
    "lawyer": "advisor",
    "legal": "policy",
    "teacher": "instructor",
    "student": "learner",
    "user": "client",
    "json": "structured",
    "answer": "respond",
    "question": "query",
    "classify": "tag",
    "extract": "select",
    "summarize": "compress",
    "tool": "utility",
    "function": "operation",
    "api": "interface",
    "private": "protected",
    "confidential": "restricted",
    "never": "avoid",
    "must": "should",
    "only": "solely",
}

PSEUDO_SNIPPETS = {
    "role_persona": [
        "You are a travel guide for short city itineraries.",
        "Act as a museum curator explaining an object.",
        "You are a friendly cooking coach.",
    ],
    "task_instruction": [
        "Generate a short slogan for a product.",
        "Translate the sentence into French.",
        "Rank the following options by price.",
    ],
    "format_spec": [
        "Return a markdown table with two columns.",
        "Use XML tags for every field.",
        "Answer as a numbered checklist.",
    ],
    "constraints": [
        "Do not mention the color blue.",
        "Never use words longer than six letters.",
        "Avoid giving any examples.",
    ],
    "context_knowledge": [
        "Use the background about a fictional island.",
        "Assume the document is about renewable energy.",
        "The passage describes a medieval trade route.",
    ],
    "tool_protocol": [
        "Call a weather API before answering.",
        "Use a calculator tool and return the expression.",
        "Follow a three-step search protocol.",
    ],
}

TOKEN_RE = re.compile(r"\{[^}]+\}|[A-Za-z0-9_./:-]+|[^\sA-Za-z0-9_./:-]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_dir", type=Path, default=Path("results/six_category_100_selection"))
    parser.add_argument("--methods", default=" ".join(METHODS))
    parser.add_argument("--categories", default=" ".join(CATEGORIES))
    parser.add_argument("--sample_n", type=int, default=100)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", default="privacybench100")
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


def chunks(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]
    if parts:
        return parts
    return [text.strip()]


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def sketch_keywords(prompt: str, limit: int = 24) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_/-]{3,}", prompt)
    stop = {
        "with", "that", "this", "from", "your", "should", "would", "there",
        "their", "about", "when", "then", "only", "must", "into", "output",
        "input", "user", "answer", "following",
    }
    seen = set()
    out = []
    for word in words:
        key = word.lower()
        if key in stop or key in seen:
            continue
        seen.add(key)
        out.append(word)
        if len(out) >= limit:
            break
    return out


def perturb_token(token: str, rng: random.Random) -> str:
    if token.startswith("{") and token.endswith("}"):
        return token
    key = token.lower().strip(".,;:!?()[]\"'")
    if key in SENSITIVE_WORDS:
        return SENSITIVE_WORDS[key]
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", token):
        if len(token) <= 3:
            return token
        mode = rng.random()
        if mode < 0.25:
            return token[:2] + "▁"
        if mode < 0.50:
            return "T" + str(abs(hash(token)) % 997)
        if mode < 0.70:
            return token[0] + "..." + token[-1]
        return token
    return token


def inferdpt_transform(prompt: str, rng: random.Random) -> tuple[str, str, str]:
    toks = TOKEN_RE.findall(prompt)
    perturbed = " ".join(perturb_token(tok, rng) for tok in toks)
    perturbed = re.sub(r"\s+", " ", perturbed).strip()
    sketch = "token-LDP perturbation; weak keywords: " + ", ".join(sketch_keywords(prompt, 18))
    return perturbed, sketch, "Token replacements approximate a black-box LDP perturbation module."


def confusionprompt_transform(prompt: str, category: str, rng: random.Random) -> tuple[str, str, str]:
    real = chunks(prompt)
    pseudo = list(PSEUDO_SNIPPETS.get(category, []))
    rng.shuffle(pseudo)
    items = []
    for idx, value in enumerate(real[:6]):
        items.append((f"R{idx+1}-{short_hash(value)}", value))
    for idx, value in enumerate(pseudo[: max(3, min(6, len(real) + 1))]):
        items.append((f"P{idx+1}-{short_hash(value)}", value))
    rng.shuffle(items)
    mixed = "\n".join(f"[{tag}] {value}" for tag, value in items)
    sketch = "mixed prompt group; genuine fragments are hidden among pseudo prompts"
    return mixed, sketch, "The attacker sees a group of real-looking sub-prompts, not the true grouping."


def split_denoise_transform(prompt: str, rng: random.Random) -> tuple[str, str, str]:
    keys = sketch_keywords(prompt, 28)
    noisy = []
    for word in keys:
        if rng.random() < 0.45:
            noisy.append(word[:2].lower() + "≈" + str(abs(hash(word)) % 100))
        else:
            noisy.append("emb_" + str(abs(hash(word)) % 1009))
    protected = " ".join(noisy)
    sketch = "noisy nearest-token embedding projection; raw text is not sent to server"
    return protected, sketch, "This approximates SnD evidence visible after client-side LDP noise on embeddings."


def confidential_prompting_transform(prompt: str, category: str, rng: random.Random) -> tuple[str, str, str]:
    real_keywords = sketch_keywords(prompt, 12)
    virtuals = []
    for i in range(4):
        pseudo = rng.choice(PSEUDO_SNIPPETS.get(category, ["You are a helpful assistant."]))
        blend = real_keywords[i::4][:3]
        suffix = " ".join(blend)
        virtuals.append(f"virtual_prompt_{i+1}: {pseudo} {suffix}".strip())
    protected = "\n".join(virtuals)
    sketch = "virtual prompts from prompt-obfuscation layer; true prompt remains hidden in confidential execution"
    return protected, sketch, "This isolates the paper's prompt-obfuscation idea, not the TEE/SMD system."


def noir_transform(prompt: str, rng: random.Random) -> tuple[str, str, str]:
    toks = [t for t in TOKEN_RE.findall(prompt) if re.fullmatch(r"\{[^}]+\}|[A-Za-z0-9_./:-]+", t)]
    mapped = []
    for tok in toks[:90]:
        if tok.startswith("{") and tok.endswith("}"):
            mapped.append(tok)
        else:
            mapped.append(f"v{abs(hash(tok)) % 5000}")
    protected = " ".join(mapped)
    sketch = "randomized tokenizer ids with placeholder preservation"
    return protected, sketch, "This approximates NOIR's local randomized tokenizer / IND vocabulary evidence."


EMOJI_ALPHABET = [
    "🔒", "🧩", "🌀", "🧭", "🧱", "🪞", "🧪", "🛡️", "📦", "🔑",
    "🌐", "📌", "⚙️", "🧬", "📝", "📎", "🎯", "🧠", "🕹️", "📡",
]


def emojicrypt_transform(prompt: str, rng: random.Random) -> tuple[str, str, str]:
    toks = TOKEN_RE.findall(prompt)
    encrypted = []
    for tok in toks[:140]:
        if tok.startswith("{") and tok.endswith("}"):
            encrypted.append(tok)
        elif re.fullmatch(r"[A-Za-z0-9_./:-]+", tok):
            digest = int(hashlib.sha256(tok.lower().encode("utf-8")).hexdigest()[:8], 16)
            encrypted.append(f"{EMOJI_ALPHABET[digest % len(EMOJI_ALPHABET)]}{digest % 997:03d}")
        else:
            encrypted.append(tok)
    protected = " ".join(encrypted)
    protected = re.sub(r"\s+", " ", protected).strip()
    sketch = "emoji-encrypted prompt; placeholders preserved; weak keywords: " + ", ".join(sketch_keywords(prompt, 12))
    return protected, sketch, "This approximates EmojiCrypt's emoji-based prompt encryption, not its full encryption stack."


DP_GENERIC_TEMPLATES = {
    "role_persona": "Use a private optimized role instruction. Preserve speaker identity only as a broad role.",
    "task_instruction": "Use a private optimized task instruction. Preserve the high-level action and expected decision.",
    "format_spec": "Use a private optimized format instruction. Preserve output shape, fields, and separators.",
    "constraints": "Use a private optimized constraint instruction. Preserve allowed and forbidden behavior at a high level.",
    "context_knowledge": "Use a private optimized context instruction. Preserve background topic without exact private facts.",
    "tool_protocol": "Use a private optimized tool protocol. Preserve procedure order and API-like requirements.",
}


def dp_opt_transform(prompt: str, category: str, rng: random.Random) -> tuple[str, str, str]:
    keys = sketch_keywords(prompt, 20)
    noisy_votes = []
    for idx, word in enumerate(keys):
        if rng.random() < 0.25:
            continue
        if rng.random() < 0.35:
            noisy_votes.append(f"vote_{idx}:{word[:2].lower()}*")
        else:
            noisy_votes.append(f"vote_{idx}:{word.lower()}")
        if len(noisy_votes) >= 10:
            break
    protected = (
        f"{DP_GENERIC_TEMPLATES.get(category, 'Use a private optimized prompt instruction.')}\n"
        f"noisy_private_votes: {', '.join(noisy_votes)}\n"
        f"dp_prompt_id: dp_{short_hash(prompt)}"
    )
    sketch = "DP ensemble prompt sketch; exact source prompt hidden behind noisy private votes"
    return protected, sketch, "This approximates DP-OPT's private prompt generation as cloud-visible noisy prompt evidence."


PLACEHOLDER_RULES = {
    "role_persona": [
        (re.compile(r"\b(doctor|lawyer|teacher|assistant|expert|advisor|coach|agent|analyst)\b", re.I), "<ROLE>"),
    ],
    "task_instruction": [
        (re.compile(r"\b(classify|extract|summarize|translate|generate|identify|rank|answer|rewrite|select)\b", re.I), "<TASK_VERB>"),
    ],
    "format_spec": [
        (re.compile(r"\b(JSON|XML|CSV|YAML|markdown|table|bullet|label|field|schema)\b", re.I), "<FORMAT>"),
    ],
    "constraints": [
        (re.compile(r"\b(do not|never|must not|only|avoid|forbidden|cannot|should not)\b", re.I), "<CONSTRAINT>"),
    ],
    "context_knowledge": [
        (re.compile(r"\b(context|background|document|passage|article|story|knowledge|scenario)\b", re.I), "<CONTEXT>"),
    ],
    "tool_protocol": [
        (re.compile(r"\b(tool|function|API|call|protocol|argument|parameter|endpoint)\b", re.I), "<TOOL_PROTOCOL>"),
    ],
}


def preempt_transform(prompt: str, category: str, rng: random.Random) -> tuple[str, str, str]:
    sanitized = prompt
    for rules in PLACEHOLDER_RULES.values():
        for pattern, repl in rules:
            sanitized = pattern.sub(repl, sanitized)
    sanitized = re.sub(r"\b[A-Z][A-Za-z0-9_-]{5,}\b", "<ENTITY>", sanitized)
    sanitized = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", sanitized)
    lines = chunks(sanitized)
    protected = "\n".join(lines[:12])
    sketch = (
        "sanitized prompt with typed placeholders; dominant category="
        + category
        + "; visible placeholders: "
        + ", ".join(sorted(set(re.findall(r"<[A-Z_]+>", protected)))[:12])
    )
    return protected, sketch, "This approximates PrEEmpt-style prompt sanitization before cloud-side model use."


def protected_representation(method: str, prompt: str, category: str, rng: random.Random) -> tuple[str, str, str]:
    if method == "confusionprompt":
        return confusionprompt_transform(prompt, category, rng)
    if method == "inferdpt":
        return inferdpt_transform(prompt, rng)
    if method == "split_denoise":
        return split_denoise_transform(prompt, rng)
    if method == "confidential_prompting":
        return confidential_prompting_transform(prompt, category, rng)
    if method == "noir":
        return noir_transform(prompt, rng)
    if method == "emojicrypt":
        return emojicrypt_transform(prompt, rng)
    if method == "dp_opt":
        return dp_opt_transform(prompt, category, rng)
    if method == "preempt":
        return preempt_transform(prompt, category, rng)
    raise ValueError(f"Unknown method: {method}")


def load_queries_and_refs(result_dir: Path, limit: int) -> tuple[list[str], list[str]]:
    queries = json.loads((result_dir / "prepared_data" / "test_data.json").read_text(encoding="utf-8"))[:limit]
    conv = json.loads((result_dir / "conventional_output.json").read_text(encoding="utf-8"))
    refs = []
    for group in conv.get("output", [])[:limit]:
        refs.append(str(group[0]) if group else "")
    return [str(q) for q in queries], refs


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    selection_dir = resolve(args.selection_dir)
    methods = [m for m in args.methods.split() if m]
    categories = [c for c in args.categories.split() if c]
    index = precomputed_index()
    summary: dict[str, dict] = {}

    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unsupported method {method}. Choices: {METHODS}")
        method_summary = {}
        for category in categories:
            rows = json.loads((selection_dir / f"{category}_selected_prompts.json").read_text(encoding="utf-8"))
            usable = [row for row in rows if row["id"] in index]
            selected = sorted(usable[: args.sample_n], key=lambda r: r["id"])
            out_dir = BASE / "results" / f"{args.output_prefix}_{method}_{category}_recovery"
            out_dir.mkdir(parents=True, exist_ok=True)
            records = []
            print(f"=== [{method}/{category}] {len(selected)} prompts ===", flush=True)
            for row in selected:
                local_rng = random.Random(f"{args.seed}:{method}:{category}:{row['id']}")
                result_dir = index[row["id"]]
                original = row.get("prompt_text") or (result_dir / "original_prompt.txt").read_text(encoding="utf-8")
                protected, semantic_sketch, extra = protected_representation(method, original, category, local_rng)
                queries, refs = load_queries_and_refs(result_dir, args.n_examples)
                examples = [
                    {
                        "user_input": q,
                        # These defenses are evaluated as high-utility protection layers.
                        # For this benchmark, the cloud-visible behavior is represented by
                        # the cached original-output behavior on matched test inputs.
                        "obfuscated_output": ref,
                        "conventional_output": ref,
                    }
                    for q, ref in zip(queries, refs)
                ]
                records.append(
                    {
                        "id": row["id"],
                        "result_dir": str(result_dir.relative_to(BASE)),
                        "prompt_label": row.get("prompt_label", ""),
                        "paper_title": row.get("paper_title", ""),
                        "scores": row.get("scores", {}),
                        "obfuscation": {
                            "method": method,
                            "evidence_label": METHOD_LABELS[method],
                            "protected_prompt_text": protected,
                            "semantic_sketch": semantic_sketch,
                            "extra_evidence": extra,
                            # Compatibility with older recovery prompt code.
                            "projection_euclidean_text": protected,
                            "projection_cosine_text": semantic_sketch,
                        },
                        "behavior_examples": examples,
                        "behavior_source": {
                            "output_tag": f"{method}_privacybench",
                            "probe_category": category,
                            "probe_set": "matched_original_test",
                            "note": "Behavior traces use cached original outputs to model a high-utility protection layer.",
                        },
                        "target_original_prompt": original,
                        "target_char_len": len(original),
                    }
                )
            (out_dir / "recovery_inputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            with (out_dir / "recovery_inputs.jsonl").open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            method_summary[category] = {"records": len(records), "output": str(out_dir.relative_to(BASE))}
        summary[method] = method_summary

    summary_path = BASE / "results" / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
