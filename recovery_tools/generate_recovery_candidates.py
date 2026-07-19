#!/usr/bin/env python3
"""Generate N candidate recovered prompts per held-out prompt via sampling.

The existing `eval_acl100_recovery_lora.py` uses greedy decoding and produces
exactly one recovered JSON per held-out prompt. For Best-of-N reranking we
need multiple candidates; this script switches the LoRA model to sampling
mode and writes a `lora_recovered_candidates_N{N}.json` file with shape:
[{id, candidates: [str, str, ...], target, target_original_prompt}, ...].
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_acl100_recovery_lora import (
    build_structured_recovery_prompt,
    parse_structured_keys,
    structured_system_message,
)
from recover_five_system_prompts import SYSTEM_MESSAGE, build_recovery_prompt


BASE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs_path", type=Path, required=True)
    p.add_argument("--lora_dir", type=Path, required=True)
    p.add_argument("--n_candidates", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=180)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_examples", type=int, default=12)
    p.add_argument("--structured_json", action="store_true", default=True)
    p.add_argument("--structured_keys", default="role,task,format,constraints,context,tool_protocol")
    p.add_argument("--target_field", default="target_structured_6d_json")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_name", default="")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def load_model(gpu: int, adapter_path: Path, model_id: str):
    cache = os.environ.get("TRANSFORMERS_CACHE", "/scratch/Hanzhi/.cache/huggingface")
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    local_gpu = 0 if visible else gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": f"cuda:{local_gpu}"},
        cache_dir=cache,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def generate_candidates(
    tok, model, record: dict,
    n_candidates: int, temperature: float, top_p: float, max_new_tokens: int,
    n_examples: int, structured_json: bool, structured_keys: list[str],
) -> list[str]:
    if structured_json:
        user_prompt = build_structured_recovery_prompt(record, n_examples, structured_keys)
        sys_msg = structured_system_message(structured_keys)
    else:
        user_prompt = build_recovery_prompt(record, mode="metadata_behavior_projection", n_examples=n_examples)
        sys_msg = SYSTEM_MESSAGE

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt},
    ]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_candidates,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return [tok.decode(out[i][ids.shape[1]:], skip_special_tokens=True).strip() for i in range(n_candidates)]


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    inputs_path = resolve(args.inputs_path)
    lora_dir = resolve(args.lora_dir)
    split_path = lora_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}; need split.json to identify held-out ids.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(split["test_ids"])
    all_records = json.loads(inputs_path.read_text(encoding="utf-8"))
    records = [r for r in all_records if r["id"] in test_ids]
    structured_keys = parse_structured_keys(args.structured_keys)

    tok, model = load_model(args.gpu, lora_dir / "adapter", args.model_id)

    out_name = args.output_name or f"lora_recovered_candidates_N{args.n_candidates}.json"
    output_path = lora_dir / out_name

    results = []
    for i, record in enumerate(records):
        print(f"[{i+1}/{len(records)}] {record['id']}: generating {args.n_candidates} candidates")
        candidates = generate_candidates(
            tok, model, record,
            n_candidates=args.n_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            n_examples=args.n_examples,
            structured_json=args.structured_json,
            structured_keys=structured_keys,
        )
        results.append({
            "id": record["id"],
            "candidates": candidates,
            "target": record.get(args.target_field, ""),
            "target_original_prompt": record.get("target_original_prompt", ""),
        })

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
