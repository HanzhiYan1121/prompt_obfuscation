#!/usr/bin/env python3
"""Evaluate the ACL100 recovery LoRA adapter on held-out prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from recover_five_system_prompts import SYSTEM_MESSAGE, build_recovery_prompt
from train_acl100_recovery_lora import (
    STRUCTURED_SYSTEM_MESSAGE,
    apply_chat_template,
    build_structured_recovery_prompt,
    parse_structured_keys,
    structured_system_message,
)
from eval_five_prompt_recovery import rouge_l_f1, token_f1, keyword_recall


BASE = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def parse_json_object(text: str) -> dict | None:
    """Parse the first complete JSON object from a recovery output."""
    start = text.find("{")
    while start >= 0:
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
                        break
        start = text.find("{", start + 1)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_path", type=Path, default=Path("results/acl100_persona_recovery/recovery_inputs.json"))
    parser.add_argument("--lora_dir", type=Path, default=Path("results/acl100_persona_recovery_lora"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Write evaluation artifacts here instead of modifying --lora_dir.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--target_field", default="target_original_prompt")
    parser.add_argument("--n_examples", type=int, default=2)
    parser.add_argument("--structured_json", action="store_true")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--structured_keys", default="role,task,constraints,format")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def load_model(gpu: int, adapter_path: Path, model_id: str):
    cache = os.environ.get("TRANSFORMERS_CACHE", "/scratch/Hanzhi/.cache/huggingface")
    visible_devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    local_gpu = 0 if visible_devices else gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": f"cuda:{local_gpu}"},
        cache_dir=cache,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate(
    tokenizer,
    model,
    record: dict,
    max_new_tokens: int,
    n_examples: int,
    structured_json: bool,
    structured_keys: list[str],
) -> str:
    user_prompt = (
        build_structured_recovery_prompt(record, n_examples, structured_keys)
        if structured_json
        else build_recovery_prompt(record, mode="metadata_behavior_projection", n_examples=n_examples)
    )
    messages = [
        {"role": "system", "content": structured_system_message(structured_keys) if structured_json else SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]
    ids = apply_chat_template(tokenizer, messages, tokenize=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    inputs_path = resolve(args.inputs_path)
    lora_dir = resolve(args.lora_dir)
    output_dir = resolve(args.output_dir) if args.output_dir is not None else lora_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((lora_dir / "split.json").read_text(encoding="utf-8"))
    test_ids = set(split["test_ids"])
    records = [r for r in json.loads(inputs_path.read_text(encoding="utf-8")) if r["id"] in test_ids]

    tokenizer, model = load_model(args.gpu, lora_dir / "adapter", args.model_id)
    structured_keys = parse_structured_keys(args.structured_keys)
    outputs = []
    eval_rows = []
    for record in records:
        print(f"Evaluating LoRA recovery: {record['id']}")
        recovered = generate(
            tokenizer,
            model,
            record,
            args.max_new_tokens,
            args.n_examples,
            args.structured_json,
            structured_keys,
        )
        ref = record[args.target_field]
        precision, recall, f1 = token_f1(recovered, ref)
        kw_recall, kw_hit, kw_miss = keyword_recall(recovered, ref)
        outputs.append(
            {
                "id": record["id"],
                "recovered_prompt": recovered,
                "target_original_prompt": record.get("target_original_prompt", ""),
                "target_field": args.target_field,
                "target": ref,
            }
        )
        eval_row = {
                "id": record["id"],
                "rougeL_f1": rouge_l_f1(recovered, ref),
                "token_precision": precision,
                "token_recall": recall,
                "token_f1": f1,
                "keyword_recall": kw_recall,
                "keywords_recovered": kw_hit,
                "keywords_missed": kw_miss,
            }
        if "executable_prompt" in structured_keys:
            recovered_obj = parse_json_object(recovered) or {}
            target_obj = parse_json_object(ref) or {}
            recovered_exec = str(recovered_obj.get("executable_prompt", ""))
            target_exec = str(target_obj.get("executable_prompt", ""))
            ep, er, ef = token_f1(recovered_exec, target_exec)
            structured_recovered = json.dumps(
                {key: recovered_obj.get(key, "") for key in structured_keys if key != "executable_prompt"},
                ensure_ascii=False,
                sort_keys=True,
            )
            structured_target = json.dumps(
                {key: target_obj.get(key, "") for key in structured_keys if key != "executable_prompt"},
                ensure_ascii=False,
                sort_keys=True,
            )
            _, _, structured_f1 = token_f1(structured_recovered, structured_target)
            eval_row.update(
                {
                    "executable_rougeL_f1": rouge_l_f1(recovered_exec, target_exec),
                    "executable_token_precision": ep,
                    "executable_token_recall": er,
                    "executable_token_f1": ef,
                    "structured_token_f1": structured_f1,
                    "dual_json_parse_success": bool(recovered_obj),
                }
            )
        eval_rows.append(eval_row)

    averages = {}
    for key in ["rougeL_f1", "token_precision", "token_recall", "token_f1", "keyword_recall"]:
        averages[key] = sum(r[key] for r in eval_rows) / len(eval_rows) if eval_rows else 0.0
    if "executable_prompt" in structured_keys:
        for key in [
            "executable_rougeL_f1",
            "executable_token_precision",
            "executable_token_recall",
            "executable_token_f1",
            "structured_token_f1",
            "dual_json_parse_success",
        ]:
            averages[key] = sum(r[key] for r in eval_rows) / len(eval_rows) if eval_rows else 0.0

    (output_dir / "lora_recovered_prompts.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "lora_recovery_eval.json").write_text(
        json.dumps({"records": eval_rows, "averages": averages}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# ACL100 LoRA Recovery Evaluation",
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| ROUGE-L F1 | {averages['rougeL_f1']:.3f} |",
        f"| Token F1 | {averages['token_f1']:.3f} |",
        f"| Keyword recall | {averages['keyword_recall']:.3f} |",
    ]
    if "executable_prompt" in structured_keys:
        lines.extend(
            [
                f"| Executable Prompt ROUGE-L F1 | {averages['executable_rougeL_f1']:.3f} |",
                f"| Executable Prompt Token F1 | {averages['executable_token_f1']:.3f} |",
                f"| Six-field Structured Token F1 | {averages['structured_token_f1']:.3f} |",
                f"| Valid Dual JSON Rate | {averages['dual_json_parse_success']:.3f} |",
            ]
        )
    lines.extend([
        "",
        "| Prompt | ROUGE-L F1 | Token F1 | Keyword Recall |",
        "|---|---:|---:|---:|",
    ])
    for row in eval_rows:
        lines.append(f"| {row['id']} | {row['rougeL_f1']:.3f} | {row['token_f1']:.3f} | {row['keyword_recall']:.3f} |")
    (output_dir / "lora_recovery_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "lora_recovery_eval_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
