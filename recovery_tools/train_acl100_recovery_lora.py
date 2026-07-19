#!/usr/bin/env python3
"""QLoRA fine-tune for ACL100 prompt recovery."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from recover_five_system_prompts import SYSTEM_MESSAGE, build_recovery_prompt
from src.finetuning_utils import CustomDataCollatorForLanguageModeling


BASE = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
STRUCTURED_SYSTEM_MESSAGE = (
    "You are a system-prompt recovery analyst. Given evidence from an obfuscated "
    "soft prompt, recover the key structure of the hidden system prompt. Output "
    "only valid JSON with keys role, task, constraints, and format. Do not add "
    "markdown or explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_path", type=Path, default=Path("results/acl100_persona_recovery/recovery_inputs.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/acl100_persona_recovery_lora"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=80)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_field", default="target_original_prompt")
    parser.add_argument("--n_examples", type=int, default=2)
    parser.add_argument("--structured_json", action="store_true")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--structured_keys", default="role,task,constraints,format")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def load_model_and_tokenizer(gpu: int, model_id: str):
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
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": f"cuda:{local_gpu}"},
        cache_dir=cache,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def parse_structured_keys(keys: str) -> list[str]:
    return [key.strip() for key in keys.split(",") if key.strip()]


def structured_system_message(keys: list[str]) -> str:
    message = (
        "You are a system-prompt recovery analyst. Given evidence from an obfuscated "
        "soft prompt, recover the key structure of the hidden system prompt. Output "
        f"only valid JSON with keys {', '.join(keys)}. Do not add markdown or explanation."
    )
    if "executable_prompt" in keys:
        message += (
            " The executable_prompt value must be a complete system prompt that can be "
            "used directly. Preserve recovered task details, labels, placeholders, output "
            "schemas, restrictions, and tool rules."
        )
    return message


def build_structured_recovery_prompt(record: dict, n_examples: int, structured_keys: list[str] | None = None) -> str:
    keys = structured_keys or parse_structured_keys("role,task,constraints,format")
    template = "{" + ", ".join(f'"{key}": "..."' for key in keys) + "}"
    base = build_recovery_prompt(record, mode="metadata_behavior_projection", n_examples=n_examples)
    instruction = (
        base
        + "\n\nReturn a JSON object with exactly these keys:\n"
        + template
        + "\n"
        + "Use empty strings when a field is not clear. Output JSON only."
    )
    if "executable_prompt" in keys:
        instruction += (
            "\nThe executable_prompt field must be a complete, directly usable system "
            "prompt reconstructed from the evidence, not a field label or commentary."
        )
    return instruction


def apply_chat_template(tokenizer, messages, *, tokenize: bool, return_tensors: str | None = None):
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": tokenize,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def make_example(
    record: dict,
    tokenizer,
    max_length: int,
    target_field: str,
    n_examples: int,
    structured_json: bool,
    structured_keys: list[str],
) -> dict:
    user_prompt = (
        build_structured_recovery_prompt(record, n_examples, structured_keys)
        if structured_json
        else build_recovery_prompt(record, mode="metadata_behavior_projection", n_examples=n_examples)
    )
    messages = [
        {"role": "system", "content": structured_system_message(structured_keys) if structured_json else SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()
    target_ids = tokenizer(record[target_field] + tokenizer.eos_token, add_special_tokens=False).input_ids
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    inputs_path = resolve(args.inputs_path)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(inputs_path.read_text(encoding="utf-8"))
    records = sorted(records, key=lambda r: r["id"])
    random.Random(args.seed).shuffle(records)
    train_records = records[: args.train_size]
    test_records = records[args.train_size :]
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_ids": [r["id"] for r in train_records],
                "test_ids": [r["id"] for r in test_records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model, tokenizer = load_model_and_tokenizer(args.gpu, args.model_id)
    structured_keys = parse_structured_keys(args.structured_keys)
    train_examples = [
        make_example(
            r,
            tokenizer,
            args.max_length,
            args.target_field,
            args.n_examples,
            args.structured_json,
            structured_keys,
        )
        for r in train_records
    ]
    train_dataset = Dataset.from_list(train_examples)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        fp16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )
    collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    config = vars(args)
    config["inputs_path"] = str(inputs_path.relative_to(BASE))
    config["output_dir"] = str(output_dir.relative_to(BASE))
    config["n_train"] = len(train_records)
    config["n_test"] = len(test_records)
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {adapter_dir.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
