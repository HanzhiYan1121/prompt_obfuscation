#!/usr/bin/env python3
"""
Multi-turn utility degradation — configurable number of turns.

Generalises eval_multi_turn.py: instead of hardcoding 3 turns, supports
an arbitrary --n_turns. History always uses conventional responses.

Usage:
    python eval_multi_turn_n.py \
        --results_dir precomputed_results/soft_pirate_truthfulqa_full \
        --n_turns 10 \
        --n_groups 16 \
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
from src.prompt_utils import zero_pad_token_attention_mask
from src.utils import set_seed

logger = logging.getLogger(__name__)
METRICS = ["cosine_similarity", "bertscore", "rouge1", "sacrebleu"]


class MultiTurnDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def format_multi_turn(tokenizer, sys_prompt, turns):
    messages = [{"role": "system", "content": sys_prompt}]
    for user_msg, asst_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        if asst_msg is not None:
            messages.append({"role": "assistant", "content": asst_msg})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def collate_fn(tokenizer):
    def fn(batch_texts):
        tok = tokenizer(batch_texts, return_tensors="pt",
                        padding=True, truncation=False)
        tok["attention_mask"] = zero_pad_token_attention_mask(
            tok["input_ids"], tok["attention_mask"], tokenizer.pad_token_id
        )
        return tok
    return fn


def generate(model_wrapper, texts, sys_tensor, placeholder_ids,
             is_soft, gen_cfg, batch_size):
    dl = DataLoader(MultiTurnDataset(texts), batch_size=batch_size,
                    collate_fn=collate_fn(model_wrapper.tokenizer), shuffle=False)
    return generate_model_responses_replace(
        model_wrapper=model_wrapper,
        dataloader=dl,
        generation_args=gen_cfg,
        sys_prompt_obf=sys_tensor,
        original_sys_prompt_ids=placeholder_ids,
        is_soft_prompt_method=is_soft,
        obf_sys_prompt_len=sys_tensor.shape[0],
        pad_token_id=model_wrapper.tokenizer.pad_token_id,
    )


def pick_best(outputs):
    return [r[0] if r else "" for r in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--n_turns",  type=int, default=10)
    parser.add_argument("--n_groups", type=int, default=16,
                        help="Number of independent conversations.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir",  default="results")
    parser.add_argument("--output_name", default="multi_turn_n_degradation.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    with open(results_dir / "params.json") as f:
        params = json.load(f)
    with open(results_dir / "generation_config.json") as f:
        gen_cfg = json.load(f)

    seed = args.seed if args.seed is not None else params.get("seed", 42)
    set_seed(seed)

    with open(results_dir / "prepared_data" / "test_data.json") as f:
        all_q = json.load(f)

    n_needed = args.n_turns * args.n_groups
    if len(all_q) < n_needed:
        args.n_groups = len(all_q) // args.n_turns
        n_needed = args.n_turns * args.n_groups
        logger.warning(f"Reduced n_groups to {args.n_groups}")

    questions = all_q[:n_needed]
    # groups[i] = list of n_turns questions for conversation i
    groups = [
        [questions[t * args.n_groups + i] for t in range(args.n_turns)]
        for i in range(args.n_groups)
    ]
    logger.info(f"n_turns={args.n_turns}, n_groups={args.n_groups}")

    # Load model
    qmode = "4bit" if params.get("quantize_4bit") else \
            "8bit" if params.get("quantize_8bit") else None
    model_wrapper = Model(params["model_name"], qmode)
    tokenizer = model_wrapper.tokenizer

    pad_tok = tokenizer.pad_token
    placeholder = f"{pad_tok}Placeholder{pad_tok}"
    placeholder_ids = tokenizer(placeholder, return_tensors="pt",
                                add_special_tokens=False).input_ids[0]

    obf_prompt = torch.load(results_dir / "best_candidate.pt", weights_only=True)
    is_soft = obf_prompt.dim() == 2
    conv_sys_ids = tokenizer(params["system_prompt"], return_tensors="pt",
                             add_special_tokens=False).input_ids[0]

    # per-sample conversation history: list of (user, assistant) pairs
    histories = [[] for _ in range(args.n_groups)]
    all_results = []

    for turn in range(1, args.n_turns + 1):
        logger.info(f"\n{'='*60}\nTurn {turn}\n{'='*60}")

        q_list = [groups[i][turn - 1] for i in range(args.n_groups)]

        # Build formatted inputs
        formatted = [
            format_multi_turn(tokenizer, placeholder, histories[i] + [(q, None)])
            for i, q in enumerate(q_list)
        ]

        logger.info("Generating conventional...")
        conv_out = generate(model_wrapper, formatted, conv_sys_ids,
                            placeholder_ids, False, gen_cfg, args.batch_size)
        logger.info("Generating obfuscated...")
        obf_out  = generate(model_wrapper, formatted, obf_prompt,
                            placeholder_ids, is_soft, gen_cfg, args.batch_size)

        scores = compute_similarity_scores(
            predictions=obf_out, references=conv_out, metric_list=METRICS
        )
        logger.info(f"Turn {turn} scores: {scores}")
        all_results.append({"turn": turn, "scores": scores})

        # Append conventional answer to history for next turn
        conv_best = pick_best(conv_out)
        for i, (q, a) in enumerate(zip(q_list, conv_best)):
            histories[i].append((q, a))

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    with open(out_path, "w") as f:
        json.dump({
            "results_dir": str(results_dir),
            "system_prompt": params["system_prompt"],
            "n_turns": args.n_turns,
            "n_groups": args.n_groups,
            "seed": seed,
            "data": all_results,
        }, f, indent=4)
    logger.info(f"Saved → {out_path}")

    print(f"\n=== Multi-Turn Degradation ({args.n_turns} turns) ===")
    print(f"{'Turn':>6} | {'Cosine':>8} | {'BERTScore':>10} | {'ROUGE-1':>8} | {'SacreBLEU':>10}")
    print("-" * 52)
    for r in all_results:
        s = r["scores"]
        print(f"{r['turn']:>6} | {s['cosine_similarity']:>8.4f} | "
              f"{s['bertscore']:>10.4f} | {s['rouge1']:>8.4f} | "
              f"{s['sacrebleu']:>10.2f}")


if __name__ == "__main__":
    setup_logging("eval_multi_turn_n.log", "INFO")
    logger = logging.getLogger(__name__)
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Done.")
