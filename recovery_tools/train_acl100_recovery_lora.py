#!/usr/bin/env python3
"""QLoRA fine-tune for ACL100 prompt recovery."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from release_config import cache_dir, LOCAL_FILES_ONLY

from recover_five_system_prompts import SYSTEM_MESSAGE, build_recovery_prompt


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
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_field", default="target_original_prompt")
    parser.add_argument("--n_examples", type=int, default=2)
    parser.add_argument("--structured_json", action="store_true")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--structured_keys", default="role,task,constraints,format")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--split_path",
        type=Path,
        default=None,
        help="Optional split.json to reuse exact train/val/test IDs instead of reshuffling.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def load_model_and_tokenizer(gpu: int, model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cache = cache_dir()
    visible_devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    local_gpu = 0 if visible_devices else gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, use_fast=False, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": f"cuda:{local_gpu}"},
        cache_dir=cache,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
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
    if "verbatim_prompt" in keys:
        message += (
            " The verbatim_prompt value must be your best literal reconstruction of the "
            "original system prompt. Preserve wording, placeholders, labels, schemas, "
            "ordering, and punctuation when the evidence supports them."
        )
    if "executable_prompt" in keys:
        message += (
            " The executable_prompt value must be a complete system prompt that can be "
            "used directly. Infer the hidden role, task, output format, hard constraints, "
            "context, and tool protocol from protected-vs-ordinary behavior differences. "
            "Do not merely summarize the evidence; write the instructions an agent should follow. "
            "When format, constraints, or tool_protocol fields are requested, preserve literal "
            "output schemas, placeholders, section labels, allowed action/tool names, and "
            "must/only/never rules instead of replacing them with generic prose."
        )
    if any(
        key in keys
        for key in ("format_contract", "constraint_checklist", "must_preserve_rules")
    ):
        message += (
            " For format_contract, recover the required response shape, schemas, field names, "
            "labels, placeholders, and ordering. For constraint_checklist, recover concrete "
            "hard rules such as must, only, never, do-not, allowed choices, and validation "
            "requirements. For must_preserve_rules, list the literal details that must survive "
            "when writing the executable prompt."
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
    if "verbatim_prompt" in keys:
        instruction += (
            "\nThe verbatim_prompt field should recover the original system prompt as "
            "literally as possible. Include placeholders, section labels, exact output "
            "schemas, and fixed wording when they can be inferred from the evidence."
        )
    if "executable_prompt" in keys:
        instruction += (
            "\nThe executable_prompt field must be a complete, directly usable system "
            "prompt reconstructed from the evidence, not a field label or commentary. "
            "Use the behavior differences to recover role, task objective, output format, "
            "constraints, private context, and tool/action protocol when present. "
            "For output-format and constraint-heavy prompts, keep exact placeholders, "
            "JSON/list schemas, section headers, allowed choices, action names, and hard "
            "rules whenever the evidence supports them."
        )
    if "format_contract" in keys:
        instruction += (
            "\nThe format_contract field should capture response format requirements: "
            "JSON schemas, required keys, section labels, placeholders, ordering, allowed "
            "values, and any fixed output template."
        )
    if "constraint_checklist" in keys:
        instruction += (
            "\nThe constraint_checklist field should capture enforceable rules: must/only/"
            "never/do-not statements, required validation conditions, forbidden behavior, "
            "and decision rules."
        )
    if "must_preserve_rules" in keys:
        instruction += (
            "\nThe must_preserve_rules field should identify exact details that must be "
            "copied into the executable prompt, especially literal field names, placeholders, "
            "labels, allowed choices, and hard constraints."
        )
    return instruction


def apply_chat_template(tokenizer, messages, *, tokenize: bool, return_tensors: str | None = None):
    import torch
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": tokenize,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not tokenize:
        return rendered
    if hasattr(rendered, "ids"):
        ids = rendered.ids
        return torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else ids
    if isinstance(rendered, list) and rendered and hasattr(rendered[0], "ids"):
        ids = [item.ids for item in rendered]
        return torch.tensor(ids, dtype=torch.long) if return_tensors == "pt" else ids
    if isinstance(rendered, dict) and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    if return_tensors == "pt" and isinstance(rendered, list):
        if rendered and isinstance(rendered[0], list):
            return torch.tensor(rendered, dtype=torch.long)
        return torch.tensor([rendered], dtype=torch.long)
    return rendered


def to_token_id_list(rendered) -> list[int]:
    import torch
    if hasattr(rendered, "input_ids"):
        return to_token_id_list(rendered.input_ids)
    if hasattr(rendered, "ids"):
        return list(rendered.ids)
    if isinstance(rendered, torch.Tensor):
        return rendered[0].tolist() if rendered.ndim > 1 else rendered.tolist()
    if isinstance(rendered, list):
        if rendered and hasattr(rendered[0], "ids"):
            return list(rendered[0].ids)
        if rendered and isinstance(rendered[0], list):
            return list(rendered[0])
        return list(rendered)
    if isinstance(rendered, dict) and "input_ids" in rendered:
        return to_token_id_list(rendered["input_ids"])
    raise TypeError(f"Unsupported chat template token output: {type(rendered)!r}")


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
    prompt_ids = to_token_id_list(
        apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            return_tensors="pt",
        )
    )
    target_ids = tokenizer(record[target_field] + tokenizer.eos_token, add_special_tokens=False).input_ids
    if len(prompt_ids) + len(target_ids) > max_length:
        if len(target_ids) >= max_length:
            target_ids = target_ids[:max_length]
            prompt_ids = []
        else:
            keep_prompt = max_length - len(target_ids)
            prompt_ids = prompt_ids[-keep_prompt:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main() -> int:
    args = parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    from src.finetuning_utils import CustomDataCollatorForLanguageModeling
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    inputs_path = resolve(args.inputs_path)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(inputs_path.read_text(encoding="utf-8"))
    records = sorted(records, key=lambda r: r["id"])
    if args.split_path is not None:
        split_path = resolve(args.split_path)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        by_id = {record["id"]: record for record in records}

        def from_split(name: str) -> list[dict]:
            missing = [prompt_id for prompt_id in split.get(name, []) if prompt_id not in by_id]
            if missing:
                raise KeyError(f"{len(missing)} IDs from {name} are missing in {inputs_path}: {missing[:3]}")
            return [by_id[prompt_id] for prompt_id in split.get(name, [])]

        train_records = from_split("train_ids")
        val_records = from_split("val_ids")
        test_records = from_split("test_ids")
    else:
        random.Random(args.seed).shuffle(records)
        train_records = records[: args.train_size]
        val_records = records[args.train_size : args.train_size + args.val_size]
        test_records = records[args.train_size + args.val_size :]
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_ids": [r["id"] for r in train_records],
                "val_ids": [r["id"] for r in val_records],
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
    val_dataset = None
    if val_records:
        val_examples = [
            make_example(
                r,
                tokenizer,
                args.max_length,
                args.target_field,
                args.n_examples,
                args.structured_json,
                structured_keys,
            )
            for r in val_records
        ]
        val_dataset = Dataset.from_list(val_examples)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "report_to": [],
        "fp16": True,
        "optim": "paged_adamw_8bit",
        "remove_unused_columns": False,
    }
    callbacks = []
    if val_dataset is not None:
        eval_strategy = "steps" if args.eval_steps > 0 else "epoch"
        training_kwargs.update(
            {
                "eval_strategy": eval_strategy,
                "save_strategy": eval_strategy,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "save_total_limit": args.save_total_limit,
            }
        )
        if args.eval_steps > 0:
            training_kwargs["eval_steps"] = args.eval_steps
            training_kwargs["save_steps"] = args.eval_steps
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            )
    training_args = TrainingArguments(**training_kwargs)
    collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    config = vars(args)
    config["inputs_path"] = str(inputs_path.relative_to(BASE))
    config["output_dir"] = str(output_dir.relative_to(BASE)) if output_dir.is_relative_to(BASE) else str(output_dir)
    if args.split_path is not None:
        split_path = resolve(args.split_path)
        config["split_path"] = str(split_path.relative_to(BASE)) if split_path.is_relative_to(BASE) else str(split_path)
    config["n_train"] = len(train_records)
    config["n_val"] = len(val_records)
    config["n_test"] = len(test_records)
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {adapter_dir.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
