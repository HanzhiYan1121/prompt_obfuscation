#!/usr/bin/env python3
"""Build higher-quality six-field recovery targets with a teacher model.

The original target builder uses overlapping regular expressions, which can copy
one sentence into several fields.  This version asks an instruction model to
perform evidence-grounded semantic decomposition, validates the result, retries
invalid annotations, and keeps the old target fields intact for ablation.

These targets are silver labels, not human gold annotations.  The output records
include validation metadata so a manually reviewed subset can be reported.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parents[1]
FIELDS = ["role", "task", "format", "constraints", "context", "tool_protocol"]
CONTENT_WORD = re.compile(r"[A-Za-z0-9_{}<>\[\]-]+")
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_. -]*\}")

SYSTEM_MESSAGE = """You create high-quality structured targets for system-prompt research.
Read the ORIGINAL SYSTEM PROMPT and return exactly one JSON object with these six string fields:

- role: the identity, persona, expertise, voice, or authority assigned to the model.
- task: the main action or decision the model must perform.
- format: the required output syntax, schema, labels, sections, ordering, or answer shape.
- constraints: mandatory rules, forbidden behavior, limits, safety rules, grounding rules, and exceptions.
- context: input variables, supplied evidence, background knowledge, examples, dialogue history, or domain setting needed to do the task.
- tool_protocol: named tools, APIs, functions, code execution, tool-call arguments, or required multi-step tool workflow.

Annotation rules:
1. Use only information supported by the original prompt. Do not invent details.
2. Describe the instruction itself. Do not treat text inside demonstrations as the instruction unless the prompt explicitly says to imitate it.
3. Keep each fact in the single best field. Do not copy the same sentence or meaning across fields.
4. Use an empty string when a field is absent. Do not fill every field by guessing.
5. Be concise, but preserve operational details such as exact labels, JSON keys, placeholders, ordering, prohibitions, tool names, and argument names.
6. Do not copy long demonstrations or candidate lists. Summarize their purpose and retain only required fixed labels or schema.
7. Every non-empty value must be a self-contained system-prompt instruction that remains grammatical and executable when the six values are concatenated without their JSON keys.
8. Write role as "Act as ..." or "You are ..."; write task as an imperative; write context as "The input includes ..." or equivalent. Do not return bare noun phrases.
9. Do not add field-name prefixes such as "Role:" or "Task:" inside values.
10. Return valid JSON only. Do not use Markdown or add commentary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=6144)
    parser.add_argument("--max_new_tokens", type=int, default=640)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--dominant_field", choices=FIELDS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def normalize(value: str) -> str:
    return " ".join(str(value).lower().split()).strip(" .;,:\"'")


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def coerce_target(raw: dict[str, Any] | None) -> dict[str, str] | None:
    if raw is None:
        return None
    target: dict[str, str] = {}
    for field in FIELDS:
        value = raw.get(field, "")
        if isinstance(value, list):
            value = "; ".join(str(item).strip() for item in value if str(item).strip())
        elif value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)
        target[field] = re.sub(r"\s+", " ", value).strip()
    return target


def duplicate_pairs(target: dict[str, str]) -> list[list[str]]:
    by_value: dict[str, list[str]] = {}
    for field in FIELDS:
        value = normalize(target[field])
        if value:
            by_value.setdefault(value, []).append(field)
    return [fields for fields in by_value.values() if len(fields) > 1]


def grounded_token_ratio(prompt: str, target: dict[str, str]) -> float:
    source = set(CONTENT_WORD.findall(prompt.lower()))
    target_tokens = [
        token
        for field in FIELDS
        for token in CONTENT_WORD.findall(target[field].lower())
        if len(token) > 2
    ]
    if not target_tokens:
        return 1.0
    return sum(token in source for token in target_tokens) / len(target_tokens)


def validate_target(
    target: dict[str, str] | None,
    prompt: str,
    dominant_field: str | None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if target is None:
        return ["invalid_json"], {"valid_json": False}
    duplicates = duplicate_pairs(target)
    if duplicates:
        errors.append("duplicate_fields")
    role = target["role"].strip()
    role_composable = not role or bool(re.match(
        r"^(?:you are|act as|serve as|behave as|adopt\b|take (?:on )?the role|respond as)",
        role,
        flags=re.I,
    ))
    source_placeholders = sorted(set(PLACEHOLDER.findall(prompt)))
    target_text = "\n".join(target.values())
    missing_placeholders = [item for item in source_placeholders if item not in target_text]
    # Very long few-shot templates can contain many demonstration variables.
    # For ordinary templates, however, dropping an input placeholder changes
    # the task and must trigger a retry.
    dominant_field_empty = bool(
        dominant_field and len(prompt.strip()) >= 20 and not target[dominant_field]
    )
    overlong = [field for field, value in target.items() if len(value) > 1200]
    if overlong:
        errors.append("overlong_fields:" + ",".join(overlong))
    nonempty = sum(bool(value) for value in target.values())
    if prompt.strip() and nonempty == 0:
        errors.append("all_fields_empty")
    metrics = {
        "valid_json": True,
        "nonempty_fields": nonempty,
        "empty_fields": [field for field in FIELDS if not target[field]],
        "duplicate_field_groups": duplicates,
        "grounded_token_ratio": round(grounded_token_ratio(prompt, target), 4),
        "field_char_lengths": {field: len(target[field]) for field in FIELDS},
        "source_placeholders": source_placeholders,
        "missing_placeholders": missing_placeholders,
        "role_composable": role_composable,
        # A category is assigned from heuristic 6D scores, so an empty dominant
        # field is evidence of category noise rather than an annotation failure.
        "dominant_field_empty": dominant_field_empty,
    }
    return errors, metrics


def repair_exact_duplicates(target: dict[str, str], dominant_field: str | None) -> dict[str, str]:
    """Remove only exact duplicates, retaining the most relevant field."""
    repaired = dict(target)
    groups = duplicate_pairs(repaired)
    for group in groups:
        keep = dominant_field if dominant_field in group else group[0]
        for field in group:
            if field != keep:
                repaired[field] = ""
    return repaired


def repair_composability(target: dict[str, str]) -> dict[str, str]:
    """Turn a supported bare role phrase into an executable role directive."""
    repaired = dict(target)
    role = repaired["role"].strip()
    if role and not re.match(
        r"^(?:you are|act as|serve as|behave as|adopt\b|take (?:on )?the role|respond as)",
        role,
        flags=re.I,
    ):
        repaired["role"] = f"Act as {role.rstrip('.')}."
    return repaired


def repair_missing_placeholders(target: dict[str, str], prompt: str) -> dict[str, str]:
    """Retain template variables without asking the teacher to infer their meaning."""
    repaired = dict(target)
    source_placeholders = sorted(set(PLACEHOLDER.findall(prompt)))
    target_text = "\n".join(repaired.values())
    missing = [item for item in source_placeholders if item not in target_text]
    if not missing or len(source_placeholders) > 12:
        return repaired

    role_vars = [
        item for item in missing if re.search(r"role|persona|speaker|profession", item, flags=re.I)
    ]
    if role_vars:
        # A generic role variable is safer than an invented concrete persona.
        repaired["role"] = "Act as " + " and ".join(role_vars) + "."
        missing = [item for item in missing if item not in role_vars]

    if missing:
        addition = "The input includes " + ", ".join(missing) + "."
        repaired["context"] = " ".join(
            part for part in [repaired["context"].strip(), addition] if part
        )
    return repaired


def has_explicit_role(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:you are|act as|serve as|your role is|imagine you are|pretend (?:that )?you are)\b|"
            r"你是|扮演|作为.{0,20}(?:助手|专家|顾问|模型)",
            prompt,
            flags=re.I,
        )
    )


def split_field_sentences(value: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", value.strip())
    return [part.strip() for part in parts if part.strip()]


def best_field_for_sentence(sentence: str, fields: list[str]) -> str:
    lower = sentence.lower()
    if "role" in fields and re.match(r"^(?:you are|act as|serve as|behave as)", lower):
        return "role"
    if "context" in fields and re.match(r"^(?:the )?input includes|^(?:you are )?(?:given|provided with)", lower):
        return "context"
    if "format" in fields and re.search(
        r"\b(?:output|return|format|json|schema|label(?:s)? only|begin|end with)\b", lower
    ):
        return "format"
    if "constraints" in fields and re.search(
        r"\b(?:must|must not|do not|don't|never|only|avoid|prefer|required|at most|no more than)\b",
        lower,
    ):
        return "constraints"
    if "tool_protocol" in fields and re.search(
        r"\b(?:tool|api|function call|protocol|workflow|execute code)\b", lower
    ):
        return "tool_protocol"
    return "task" if "task" in fields else fields[0]


def remove_cross_field_sentence_duplicates(target: dict[str, str]) -> dict[str, str]:
    """Keep an exactly repeated instruction sentence in only its best field."""
    repaired = dict(target)
    occurrences: dict[str, list[tuple[str, str]]] = {}
    for field in FIELDS:
        for sentence in split_field_sentences(repaired[field]):
            occurrences.setdefault(normalize(sentence), []).append((field, sentence))
    remove: dict[str, set[str]] = {field: set() for field in FIELDS}
    for normalized, items in occurrences.items():
        distinct_fields = list(dict.fromkeys(field for field, _ in items))
        if not normalized or len(distinct_fields) < 2:
            continue
        keep = best_field_for_sentence(items[0][1], distinct_fields)
        for field in distinct_fields:
            if field != keep:
                remove[field].add(normalized)
    for field in FIELDS:
        kept = [
            sentence
            for sentence in split_field_sentences(repaired[field])
            if normalize(sentence) not in remove[field]
        ]
        repaired[field] = " ".join(kept).strip()
    return repaired


def repair_sparse_question_template(target: dict[str, str], prompt: str) -> dict[str, str]:
    """Avoid semantic invention for short question/answer-only templates."""
    compact = " ".join(prompt.split())
    match = re.match(r"^(\{question[^}]*\})(.*)$", compact, flags=re.I)
    if not match or len(compact) > 180 or has_explicit_role(prompt):
        return target
    placeholder, suffix = match.groups()
    repaired = {field: "" for field in FIELDS}
    repaired["task"] = f"Answer {placeholder}."
    repaired["context"] = f"The input includes {placeholder}."
    if suffix.strip():
        repaired["format"] = f"Use this answer template: {suffix.strip()}"
    return repaired


def repair_short_prompt_overreach(target: dict[str, str], prompt: str) -> dict[str, str]:
    """Do not let the teacher invent semantics for a short standalone instruction."""
    compact = " ".join(prompt.split()).strip()
    if (
        not compact
        or len(compact) > 140
        or has_explicit_role(prompt)
        or re.search(r"\b(?:User|Assistant|System)\s*:", prompt, flags=re.I)
    ):
        return target
    # The dedicated question-template repair gives a cleaner task/format split.
    if re.match(r"^\{question[^}]*\}", compact, flags=re.I):
        return target

    repaired = {field: "" for field in FIELDS}
    if re.match(
        r"^(?:only (?:answer|output|return)|output|return|respond with|reply with)",
        compact,
        flags=re.I,
    ):
        repaired["format"] = compact
    else:
        repaired["task"] = compact
    return repaired


def finalize_target(
    target: dict[str, str], prompt: str, dominant_field: str | None
) -> dict[str, str]:
    repaired = repair_exact_duplicates(target, dominant_field)
    if not has_explicit_role(prompt):
        repaired["role"] = ""
    repaired = repair_sparse_question_template(repaired, prompt)
    repaired = repair_short_prompt_overreach(repaired, prompt)
    repaired = repair_composability(repaired)
    repaired = repair_missing_placeholders(repaired, prompt)
    repaired = remove_cross_field_sentence_duplicates(repaired)
    return repaired


def user_message(prompt: str, retry_errors: list[str] | None = None) -> str:
    message = f"ORIGINAL SYSTEM PROMPT:\n<original_prompt>\n{prompt}\n</original_prompt>"
    if retry_errors:
        message += (
            "\n\nYour previous annotation failed validation: "
            + ", ".join(retry_errors)
            + ". Return a corrected JSON object only."
        )
    return message


def encode_messages(tokenizer, prompts: list[str], max_input_tokens: int) -> dict[str, torch.Tensor]:
    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    return tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    encoded = encode_messages(tokenizer, prompts, args.max_input_tokens)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_length = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)


def save_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_path.parent / "recovery_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(records: list[dict[str, Any]], output_path: Path) -> None:
    annotations = [r.get("target_structured_6d_v2_validation", {}) for r in records]
    error_counts = Counter(error for item in annotations for error in item.get("errors", []))
    nonempty = Counter()
    for record in records:
        target = record.get("target_structured_6d_v2", {})
        for field in FIELDS:
            nonempty[field] += bool(target.get(field, ""))
    ratios = [item.get("grounded_token_ratio", 0.0) for item in annotations if item]
    report = {
        "records": len(records),
        "teacher_model": records[0].get("target_structured_6d_v2_meta", {}).get("teacher_model") if records else None,
        "dominant_field": records[0].get("target_structured_6d_v2_meta", {}).get("dominant_field") if records else None,
        "records_with_validation_errors": sum(bool(item.get("errors")) for item in annotations),
        "validation_error_counts": dict(error_counts),
        "dominant_field_empty_count": sum(
            bool(item.get("dominant_field_empty")) for item in annotations
        ),
        "nonempty_field_counts": dict(nonempty),
        "mean_grounded_token_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
    }
    (output_path.parent / "target_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    source_records = json.loads(input_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        source_records = source_records[: args.limit]

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for record in json.loads(output_path.read_text(encoding="utf-8")):
            validation = record.get("target_structured_6d_v2_validation", {})
            if record.get("target_structured_6d_v2_json") and not validation.get("errors"):
                completed[str(record.get("id"))] = record

    records = [completed.get(str(record.get("id")), dict(record)) for record in source_records]
    pending = [i for i, record in enumerate(records) if not record.get("target_structured_6d_v2_json")]
    if not pending:
        write_report(records, output_path)
        print(f"No pending records; validated {len(records)} records in {output_path.relative_to(BASE)}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side="left",
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": args.gpu},
        local_files_only=args.local_files_only,
    )
    model.eval()

    processed = 0
    for start in range(0, len(pending), args.batch_size):
        indices = pending[start : start + args.batch_size]
        messages = [user_message(records[index]["target_original_prompt"]) for index in indices]
        outputs = generate_batch(model, tokenizer, messages, args)

        for index, raw_output in zip(indices, outputs):
            prompt = records[index]["target_original_prompt"]
            target = coerce_target(extract_json_object(raw_output))
            errors, metrics = validate_target(target, prompt, args.dominant_field)
            retries = 0
            while errors and retries < args.max_retries:
                retries += 1
                retry_output = generate_batch(
                    model,
                    tokenizer,
                    [user_message(prompt, errors)],
                    args,
                )[0]
                raw_output = retry_output
                target = coerce_target(extract_json_object(raw_output))
                errors, metrics = validate_target(target, prompt, args.dominant_field)

            if target is None:
                target = {field: "" for field in FIELDS}
            target = finalize_target(target, prompt, args.dominant_field)
            final_errors, final_metrics = validate_target(target, prompt, args.dominant_field)
            records[index]["target_structured_6d_v2"] = target
            records[index]["target_structured_6d_v2_json"] = json.dumps(
                target, ensure_ascii=False, sort_keys=True
            )
            records[index]["target_structured_6d_v2_validation"] = {
                **final_metrics,
                "errors": final_errors,
                "retries": retries,
            }
            records[index]["target_structured_6d_v2_meta"] = {
                "teacher_model": args.model_id,
                "dominant_field": args.dominant_field,
                "method": "teacher_semantic_decomposition_v2",
            }
            processed += 1

        if processed % args.checkpoint_every < len(indices):
            save_records(records, output_path)
            print(f"Checkpoint: {processed}/{len(pending)} newly annotated records", flush=True)

    save_records(records, output_path)
    write_report(records, output_path)
    print(f"Saved {len(records)} v2 structured targets to {output_path.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
