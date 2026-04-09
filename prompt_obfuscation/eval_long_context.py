#!/usr/bin/env python3
"""
Evaluate utility degradation as user message length increases.

Hypothesis: As the user message grows longer, the 10-token soft prompt's
relative influence shrinks, causing obfuscated outputs to diverge from
conventional outputs.

Design:
- Load best_candidate.pt from a results_dir
- Prepend N tokens of neutral filler to each test question
- Context lengths tested: 0 (baseline), 128, 256, 512, 1024 extra tokens
- Generate with conventional and obfuscated prompts at each length
- Compare metrics: cosine_similarity, bertscore, rouge1, sacrebleu

Usage:
    python eval_long_context.py \
        --results_dir precomputed_results/soft_pirate_truthfulqa_full \
        --context_lengths 0 128 256 512 1024 \
        --n_samples 40 \
        --batch_size 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import generate_model_responses_replace, generate_model_responses
from src.output_similarity import compute_similarity_scores
from src.prompt_utils import apply_chat_template_to_batch, zero_pad_token_attention_mask
from src.utils import set_seed

logger = logging.getLogger(__name__)

# Neutral filler sentences — topically neutral, won't bias the answer
FILLER_SENTENCE = (
    "This document provides background context for reference purposes only. "
    "The following information is general knowledge and should not be taken as specific advice. "
    "Please read the question carefully before formulating your response. "
    "Additional context is included here to provide a comprehensive overview of the subject matter. "
    "This text does not contain the answer and is provided solely as supplementary material. "
)


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def create_collate_fn(tokenizer, system_prompt: str):
    def collate_fn(batch_texts: list[str]):
        formatted = apply_chat_template_to_batch(
            user_prompts=batch_texts,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
        tokenized = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        tokenized["attention_mask"] = zero_pad_token_attention_mask(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenized
    return collate_fn


def build_padded_questions(
    questions: list[str],
    tokenizer,
    target_extra_tokens: int,
) -> list[str]:
    """Prepend filler text of ~target_extra_tokens to each question."""
    if target_extra_tokens == 0:
        return questions

    # Build filler until long enough, then trim to exact token count
    filler = FILLER_SENTENCE
    while len(tokenizer.encode(filler, add_special_tokens=False)) < target_extra_tokens:
        filler += FILLER_SENTENCE

    filler_ids = tokenizer.encode(filler, add_special_tokens=False)[:target_extra_tokens]
    filler_text = tokenizer.decode(filler_ids, skip_special_tokens=True)
    actual_len = len(filler_ids)
    logger.info(f"Filler length: {actual_len} tokens (target: {target_extra_tokens})")

    padded = [f"{filler_text}\n\n{q}" for q in questions]
    return padded


def run_generation(
    model_wrapper: Model,
    questions: list[str],
    sys_prompt_tensor: torch.Tensor,
    placeholder_sys_prompt: str,
    placeholder_sys_ids: torch.Tensor,
    is_soft: bool,
    generation_config: dict,
    batch_size: int,
) -> list[list[str]]:
    """Generate responses using the given system prompt (conventional or obfuscated)."""
    dataset = TextDataset(questions)
    collate_fn = create_collate_fn(model_wrapper.tokenizer, placeholder_sys_prompt)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return generate_model_responses_replace(
        model_wrapper=model_wrapper,
        dataloader=dataloader,
        generation_args=generation_config,
        sys_prompt_obf=sys_prompt_tensor,
        original_sys_prompt_ids=placeholder_sys_ids,
        is_soft_prompt_method=is_soft,
        obf_sys_prompt_len=sys_prompt_tensor.shape[0],
        pad_token_id=model_wrapper.tokenizer.pad_token_id,
    )


def main():
    parser = argparse.ArgumentParser(description="Long-context utility degradation experiment.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to precomputed_results/XXX directory.")
    parser.add_argument("--context_lengths", type=int, nargs="+",
                        default=[0, 128, 256, 512, 1024],
                        help="Extra tokens to prepend at each test point.")
    parser.add_argument("--n_samples", type=int, default=40,
                        help="Number of test questions to use.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_name", type=str, default="long_context_degradation.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load params and config
    with open(results_dir / "params.json") as f:
        params = json.load(f)
    with open(results_dir / "generation_config.json") as f:
        generation_config = json.load(f)

    seed = args.seed if args.seed is not None else params.get("seed", 42)
    set_seed(seed)

    # Load test data
    test_data_path = results_dir / "prepared_data" / "test_data.json"
    with open(test_data_path) as f:
        all_questions = json.load(f)
    questions = all_questions[: args.n_samples]
    logger.info(f"Using {len(questions)} test questions from {test_data_path}")

    # Load model
    quantization_mode = None
    if params.get("quantize_4bit", False):
        quantization_mode = "4bit"
    elif params.get("quantize_8bit", False):
        quantization_mode = "8bit"
    model_wrapper = Model(params["model_name"], quantization_mode)
    logger.info(f"Loaded model: {params['model_name']}")

    pad_token_str = model_wrapper.tokenizer.pad_token
    placeholder_sys_prompt = f"{pad_token_str}Placeholder{pad_token_str}"
    placeholder_sys_ids = model_wrapper.tokenizer(
        placeholder_sys_prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    # Load obfuscated prompt (soft embedding)
    best_candidate_path = results_dir / "best_candidate.pt"
    obf_prompt = torch.load(best_candidate_path, weights_only=True)
    is_soft = obf_prompt.dim() == 2
    logger.info(f"Loaded {'soft' if is_soft else 'hard'} obfuscated prompt: shape {obf_prompt.shape}")

    # Build conventional system prompt token IDs
    conv_sys_prompt_str = params["system_prompt"]
    conv_sys_ids = model_wrapper.tokenizer(
        conv_sys_prompt_str, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    METRICS = ["cosine_similarity", "bertscore", "rouge1", "sacrebleu"]

    all_results = []

    for ctx_len in args.context_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Context length: {ctx_len} extra tokens")
        logger.info(f"{'='*60}")

        padded_questions = build_padded_questions(questions, model_wrapper.tokenizer, ctx_len)

        # --- Conventional ---
        logger.info("Generating with conventional prompt...")
        conv_outputs = run_generation(
            model_wrapper, padded_questions,
            sys_prompt_tensor=conv_sys_ids,
            placeholder_sys_prompt=placeholder_sys_prompt,
            placeholder_sys_ids=placeholder_sys_ids,
            is_soft=False,
            generation_config=generation_config,
            batch_size=args.batch_size,
        )

        # --- Obfuscated ---
        logger.info("Generating with obfuscated prompt...")
        obf_outputs = run_generation(
            model_wrapper, padded_questions,
            sys_prompt_tensor=obf_prompt,
            placeholder_sys_prompt=placeholder_sys_prompt,
            placeholder_sys_ids=placeholder_sys_ids,
            is_soft=is_soft,
            generation_config=generation_config,
            batch_size=args.batch_size,
        )

        # --- Compute metrics ---
        logger.info("Computing similarity metrics...")
        scores = compute_similarity_scores(
            predictions=obf_outputs,
            references=conv_outputs,
            metric_list=METRICS,
        )
        logger.info(f"Scores at ctx_len={ctx_len}: {scores}")

        all_results.append({
            "extra_context_tokens": ctx_len,
            "scores": scores,
            "n_samples": len(questions),
        })

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    with open(output_path, "w") as f:
        json.dump({
            "results_dir": str(results_dir),
            "system_prompt": params["system_prompt"],
            "obf_sys_prompt_len": params.get("obf_sys_prompt_len", 10),
            "n_samples": len(questions),
            "seed": seed,
            "data": all_results,
        }, f, indent=4)

    logger.info(f"\nSaved results to {output_path}")

    # Print summary table
    print("\n=== Long Context Degradation Results ===")
    print(f"{'ctx_tokens':>12} | {'cosine':>8} | {'bertscore':>10} | {'rouge1':>8} | {'sacrebleu':>10}")
    print("-" * 58)
    for r in all_results:
        s = r["scores"]
        print(
            f"{r['extra_context_tokens']:>12} | "
            f"{s.get('cosine_similarity', float('nan')):>8.4f} | "
            f"{s.get('bertscore', float('nan')):>10.4f} | "
            f"{s.get('rouge1', float('nan')):>8.4f} | "
            f"{s.get('sacrebleu', float('nan')):>10.2f}"
        )


if __name__ == "__main__":
    setup_logging("eval_long_context.log", "INFO")
    logger = logging.getLogger(__name__)
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Done.")
