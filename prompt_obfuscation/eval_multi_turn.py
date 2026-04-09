#!/usr/bin/env python3
"""
Evaluate utility degradation across multi-turn conversations.

Hypothesis: The 10-token soft prompt is optimized for single-turn Q&A.
As conversation history grows (more turns), the system prompt's relative
influence over the model's output should diminish, causing the obfuscated
outputs to diverge from the conventional outputs at later turns.

Design:
- Group test questions into triplets (Q1, Q2, Q3)
- For each turn t in {1, 2, 3}:
    - Build input: [sys] [Q1] [conv_A1] [Q2] [conv_A2] ... [Qt]
      (previous turns always use conventional responses to control for history quality)
    - Generate conv_At and obf_At
    - Compute sim(conv_At, obf_At)
- Hypothesis: similarity decreases as t increases

Usage:
    python eval_multi_turn.py \
        --results_dir precomputed_results/soft_pirate_truthfulqa_full \
        --n_triplets 30 \
        --batch_size 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import generate_model_responses_replace
from src.output_similarity import compute_similarity_scores
from src.prompt_utils import (
    find_sys_prompt_indices_batch,
    replace_sys_prompt_batch,
    update_attention_mask_batch,
    zero_pad_token_attention_mask,
)
from src.utils import set_seed

logger = logging.getLogger(__name__)

METRICS = ["cosine_similarity", "bertscore", "rouge1", "sacrebleu"]


class MultiTurnDataset(Dataset):
    """Dataset of pre-formatted multi-turn conversation strings."""
    def __init__(self, formatted_texts: list[str]):
        self.texts = formatted_texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def format_multi_turn(
    tokenizer: PreTrainedTokenizerBase,
    sys_prompt: str,
    turns: list[tuple[str, str | None]],
) -> str:
    """
    Format a multi-turn conversation using the tokenizer's chat template.

    Args:
        tokenizer: The tokenizer (must support apply_chat_template).
        sys_prompt: System prompt content.
        turns: List of (user_message, assistant_response) pairs.
               The last pair should have assistant_response=None to trigger generation.

    Returns:
        Formatted string ready for tokenization.
    """
    messages = [{"role": "system", "content": sys_prompt}]
    for i, (user_msg, asst_msg) in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})
        if asst_msg is not None:
            messages.append({"role": "assistant", "content": asst_msg})

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def create_multi_turn_collate_fn(tokenizer: PreTrainedTokenizerBase):
    """Collate function for already-formatted multi-turn strings."""
    def collate_fn(batch_texts: list[str]):
        tokenized = tokenizer(
            batch_texts,
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


def generate_from_formatted(
    model_wrapper: Model,
    formatted_texts: list[str],
    sys_prompt_tensor: torch.Tensor,
    placeholder_sys_ids: torch.Tensor,
    is_soft: bool,
    generation_config: dict,
    batch_size: int,
) -> list[list[str]]:
    """
    Generate responses from pre-formatted multi-turn texts, replacing the
    placeholder system prompt with either conv or obf prompt.
    """
    dataset = MultiTurnDataset(formatted_texts)
    collate_fn = create_multi_turn_collate_fn(model_wrapper.tokenizer)
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


def pick_best_response(output: list[list[str]]) -> list[str]:
    """Pick the first generated response per sample (used for building history)."""
    return [responses[0] if responses else "" for responses in output]


def main():
    parser = argparse.ArgumentParser(description="Multi-turn utility degradation experiment.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to precomputed_results/XXX directory.")
    parser.add_argument("--n_triplets", type=int, default=30,
                        help="Number of question triplets to evaluate.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_name", type=str, default="multi_turn_degradation.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    with open(results_dir / "params.json") as f:
        params = json.load(f)
    with open(results_dir / "generation_config.json") as f:
        generation_config = json.load(f)

    seed = args.seed if args.seed is not None else params.get("seed", 42)
    set_seed(seed)

    # Load test questions
    test_data_path = results_dir / "prepared_data" / "test_data.json"
    with open(test_data_path) as f:
        all_questions = json.load(f)

    # We need 3 * n_triplets questions; each triplet forms a 3-turn conversation
    n_needed = 3 * args.n_triplets
    if len(all_questions) < n_needed:
        logger.warning(
            f"Only {len(all_questions)} questions available; "
            f"reducing n_triplets to {len(all_questions) // 3}"
        )
        args.n_triplets = len(all_questions) // 3
        n_needed = 3 * args.n_triplets

    questions = all_questions[:n_needed]
    # Group into triplets: Q1, Q2, Q3 for each conversation
    triplets = [(questions[i], questions[i+1], questions[i+2])
                for i in range(0, n_needed, 3)]
    logger.info(f"Using {len(triplets)} question triplets")

    # Load model
    quantization_mode = None
    if params.get("quantize_4bit", False):
        quantization_mode = "4bit"
    elif params.get("quantize_8bit", False):
        quantization_mode = "8bit"
    model_wrapper = Model(params["model_name"], quantization_mode)
    logger.info(f"Loaded model: {params['model_name']}")

    tokenizer = model_wrapper.tokenizer
    pad_token_str = tokenizer.pad_token
    # placeholder_sys_prompt includes pad tokens as delimiters so that
    # find_sys_prompt_indices can locate and replace the system prompt slot.
    placeholder_sys_prompt = f"{pad_token_str}Placeholder{pad_token_str}"
    placeholder_sys_ids = tokenizer(
        placeholder_sys_prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    # Obfuscated prompt
    obf_prompt = torch.load(results_dir / "best_candidate.pt", weights_only=True)
    is_soft = obf_prompt.dim() == 2
    logger.info(f"Loaded {'soft' if is_soft else 'hard'} obfuscated prompt: shape {obf_prompt.shape}")

    # Conventional prompt token IDs
    conv_sys_str = params["system_prompt"]
    conv_sys_ids = tokenizer(
        conv_sys_str, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    all_results = []

    # -------------------------
    # TURN 1: simple single-turn (baseline — should match paper's existing results)
    # -------------------------
    logger.info("\n" + "="*60)
    logger.info("TURN 1 (single-turn baseline)")
    logger.info("="*60)

    q1_list = [t[0] for t in triplets]  # just the first question of each triplet

    # Format turn-1 inputs with placeholder sys prompt (including pad-token delimiters)
    turn1_formatted = [
        format_multi_turn(tokenizer, placeholder_sys_prompt, [(q, None)])
        for q in q1_list
    ]

    conv_A1_outputs = generate_from_formatted(
        model_wrapper, turn1_formatted,
        sys_prompt_tensor=conv_sys_ids,
        placeholder_sys_ids=placeholder_sys_ids,
        is_soft=False,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )
    obf_A1_outputs = generate_from_formatted(
        model_wrapper, turn1_formatted,
        sys_prompt_tensor=obf_prompt,
        placeholder_sys_ids=placeholder_sys_ids,
        is_soft=is_soft,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )

    scores_t1 = compute_similarity_scores(
        predictions=obf_A1_outputs,
        references=conv_A1_outputs,
        metric_list=METRICS,
    )
    logger.info(f"Turn 1 scores: {scores_t1}")
    all_results.append({"turn": 1, "scores": scores_t1})

    # -------------------------
    # TURN 2: [Q1, conv_A1, Q2]
    # -------------------------
    logger.info("\n" + "="*60)
    logger.info("TURN 2 ([Q1, conv_A1, Q2])")
    logger.info("="*60)

    # Use the first generated response from turn 1 (conventional) as history
    conv_A1_best = pick_best_response(conv_A1_outputs)
    q2_list = [t[1] for t in triplets]

    turn2_formatted = [
        format_multi_turn(tokenizer, placeholder_sys_prompt, [
            (q1, a1_conv),
            (q2, None),
        ])
        for (q1, a1_conv, q2) in zip(q1_list, conv_A1_best, q2_list)
    ]

    conv_A2_outputs = generate_from_formatted(
        model_wrapper, turn2_formatted,
        sys_prompt_tensor=conv_sys_ids,
        placeholder_sys_ids=placeholder_sys_ids,
        is_soft=False,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )
    obf_A2_outputs = generate_from_formatted(
        model_wrapper, turn2_formatted,
        sys_prompt_tensor=obf_prompt,
        placeholder_sys_ids=placeholder_sys_ids,
        is_soft=is_soft,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )

    scores_t2 = compute_similarity_scores(
        predictions=obf_A2_outputs,
        references=conv_A2_outputs,
        metric_list=METRICS,
    )
    logger.info(f"Turn 2 scores: {scores_t2}")
    all_results.append({"turn": 2, "scores": scores_t2})

    # -------------------------
    # TURN 3: [Q1, conv_A1, Q2, conv_A2, Q3]
    # -------------------------
    logger.info("\n" + "="*60)
    logger.info("TURN 3 ([Q1, conv_A1, Q2, conv_A2, Q3])")
    logger.info("="*60)

    conv_A2_best = pick_best_response(conv_A2_outputs)
    q3_list = [t[2] for t in triplets]

    turn3_formatted = [
        format_multi_turn(tokenizer, placeholder_sys_prompt, [
            (q1, a1_conv),
            (q2, a2_conv),
            (q3, None),
        ])
        for (q1, a1_conv, q2, a2_conv, q3) in zip(
            q1_list, conv_A1_best, q2_list, conv_A2_best, q3_list
        )
    ]

    conv_A3_outputs = generate_from_formatted(
        model_wrapper, turn3_formatted,
        sys_prompt_tensor=conv_sys_ids,
        placeholder_sys_ids=placeholder_sys_ids,
        is_soft=False,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )
    obf_A3_outputs = generate_from_formatted(
        model_wrapper, turn3_formatted,
        sys_prompt_tensor=obf_prompt,
        placeholder_sys_ids=placeholder_sys_ids,
        is_soft=is_soft,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )

    scores_t3 = compute_similarity_scores(
        predictions=obf_A3_outputs,
        references=conv_A3_outputs,
        metric_list=METRICS,
    )
    logger.info(f"Turn 3 scores: {scores_t3}")
    all_results.append({"turn": 3, "scores": scores_t3})

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    save_data = {
        "results_dir": str(results_dir),
        "system_prompt": params["system_prompt"],
        "obf_sys_prompt_len": params.get("obf_sys_prompt_len", 10),
        "n_triplets": len(triplets),
        "seed": seed,
        "data": all_results,
        "sample_triplets": [
            {"Q1": q1_list[i], "Q2": q2_list[i], "Q3": q3_list[i]}
            for i in range(min(3, len(q1_list)))
        ],
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=4)
    logger.info(f"\nSaved results to {output_path}")

    # Print summary table
    print("\n=== Multi-Turn Degradation Results ===")
    print(f"{'turn':>6} | {'cosine':>8} | {'bertscore':>10} | {'rouge1':>8} | {'sacrebleu':>10}")
    print("-" * 52)
    for r in all_results:
        s = r["scores"]
        print(
            f"{r['turn']:>6} | "
            f"{s.get('cosine_similarity', float('nan')):>8.4f} | "
            f"{s.get('bertscore', float('nan')):>10.4f} | "
            f"{s.get('rouge1', float('nan')):>8.4f} | "
            f"{s.get('sacrebleu', float('nan')):>10.2f}"
        )


if __name__ == "__main__":
    setup_logging("eval_multi_turn.log", "INFO")
    logger = logging.getLogger(__name__)
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Done.")
