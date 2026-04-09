"""
L4 AirSim Drone – Automated Evaluation Pipeline
=================================================
Runs each test case against:
  (A) Conventional system prompt (baseline)
  (B) Obfuscated soft prompt (from precomputed_results/L4_sentiment)

Scores each output with DroneCodeEvaluator and writes:
  results/l4_drone_eval.json   – full per-case data
  results/l4_drone_eval.csv    – summary table

Usage:
    python run_l4_evaluation.py [--gpu 0] [--result_dir precomputed_results/L4_sentiment]
"""

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

import torch

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from l4_drone_evaluator import DroneCodeEvaluator
from l4_drone_testcases import TEST_CASES

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
RESULT_DIR = BASE / "precomputed_results" / "L4_sentiment"
OUT_DIR    = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── system prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_API = (BASE / "robot system prompt/system_prompts/airsim_drone_API.txt").read_text()
SYSTEM_PROMPT_RULES = (BASE / "robot system prompt/system_prompts/airsim_drone.txt").read_text()
FULL_SYSTEM_PROMPT = SYSTEM_PROMPT_RULES + "\n\n" + SYSTEM_PROMPT_API


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(gpu: int):
    """Load the base 4-bit Llama model (conventional mode)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/scratch/Hanzhi/.cache/huggingface")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map={"": f"cuda:{gpu}"},
        cache_dir=HF_CACHE,
    )
    model.eval()
    return tokenizer, model


def load_obfuscated_prompt(result_dir: Path, device: str):
    """Load the best soft-prompt embedding tensor."""
    pt_path = result_dir / "best_candidate.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No best_candidate.pt in {result_dir}")
    return torch.load(pt_path, map_location=device)


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

GEN_KWARGS = dict(max_new_tokens=300, temperature=0.0, do_sample=False)


def generate_conventional(query: str, tokenizer, model) -> str:
    """Generate with the original readable system prompt."""
    messages = [
        {"role": "system", "content": FULL_SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids, **GEN_KWARGS)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def generate_obfuscated(query: str, tokenizer, model, soft_embed) -> str:
    """
    Generate with the obfuscated soft prompt.
    The soft prompt embedding replaces the system prompt token embeddings.
    """
    from transformers import DynamicCache

    device = model.device
    embed_layer = model.get_input_embeddings()

    # User turn only (no system prompt in token space)
    user_msgs = [{"role": "user", "content": query}]
    input_ids = tokenizer.apply_chat_template(
        user_msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    user_embeds = embed_layer(input_ids)  # [1, seq, hidden]

    # Prepend soft prompt embeddings
    soft = soft_embed.to(device).unsqueeze(0)  # [1, n_tokens, hidden]
    combined = torch.cat([soft, user_embeds], dim=1)  # [1, n_soft+seq, hidden]

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            **GEN_KWARGS,
            use_cache=True,
        )
    # Decode only the newly generated tokens
    n_input_tokens = combined.shape[1]
    return tokenizer.decode(out[0][n_input_tokens:], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(gpu: int, result_dir: Path, n_cases: int | None = None, tag: str = ""):
    device = f"cuda:{gpu}"
    evaluator = DroneCodeEvaluator()

    print(f"Loading model on {device}...")
    tokenizer, model = load_model(gpu)

    print(f"Loading soft prompt from {result_dir}...")
    soft_embed = load_obfuscated_prompt(result_dir, device)

    cases = TEST_CASES[:n_cases] if n_cases else TEST_CASES
    records = []

    for tc in cases:
        qid    = tc["id"]
        query  = tc["query"]
        tier   = tc["tier"]
        gt     = tc["expected_states"]

        print(f"\n{'─'*60}")
        print(f"[{qid}] ({tier}) {query[:60]}...")

        # ── Conventional ──────────────────────────────────────────────────
        conv_code = generate_conventional(query, tokenizer, model)
        conv_res  = evaluator.evaluate(conv_code, gt)
        print(f"  Conv  → fmt={conv_res.format_score:.2f}  "
              f"api={conv_res.api_score:.2f}  "
              f"traj={conv_res.trajectory_score}")

        # ── Obfuscated ────────────────────────────────────────────────────
        obf_code = generate_obfuscated(query, tokenizer, model, soft_embed)
        obf_res  = evaluator.evaluate(obf_code, gt)
        print(f"  Obf   → fmt={obf_res.format_score:.2f}  "
              f"api={obf_res.api_score:.2f}  "
              f"traj={obf_res.trajectory_score}")

        record = {
            "id":   qid,
            "tier": tier,
            "query": query,
            # Conventional
            "conv_code":           conv_code,
            "conv_format":         conv_res.format_score,
            "conv_api":            conv_res.api_score,
            "conv_trajectory":     conv_res.trajectory_score,
            "conv_composite":      conv_res.composite_score,
            "conv_format_viols":   conv_res.format_detail.violations,
            "conv_api_viols":      conv_res.api_detail.violations,
            "conv_traj_viols":     conv_res.trajectory_detail.violations,
            "conv_sim_states":     conv_res.trajectory_detail.simulated_states,
            # Obfuscated
            "obf_code":            obf_code,
            "obf_format":          obf_res.format_score,
            "obf_api":             obf_res.api_score,
            "obf_trajectory":      obf_res.trajectory_score,
            "obf_composite":       obf_res.composite_score,
            "obf_format_viols":    obf_res.format_detail.violations,
            "obf_api_viols":       obf_res.api_detail.violations,
            "obf_traj_viols":      obf_res.trajectory_detail.violations,
            "obf_sim_states":      obf_res.trajectory_detail.simulated_states,
            # Delta
            "delta_format":    round(conv_res.format_score - obf_res.format_score, 4),
            "delta_api":       round(conv_res.api_score    - obf_res.api_score,    4),
            "delta_trajectory": round(
                (conv_res.trajectory_score or 0) - (obf_res.trajectory_score or 0), 4
            ),
        }
        records.append(record)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    suffix = f"_{tag}" if tag else ""
    json_path = OUT_DIR / f"l4_drone_eval{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved full results → {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / f"l4_drone_eval{suffix}.csv"
    csv_fields = [
        "id", "tier",
        "conv_format", "conv_api", "conv_trajectory", "conv_composite",
        "obf_format",  "obf_api",  "obf_trajectory",  "obf_composite",
        "delta_format", "delta_api", "delta_trajectory",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in csv_fields})
    print(f"Saved CSV summary    → {csv_path}")

    # ── Print aggregate table ──────────────────────────────────────────────────
    print_aggregate(records)

    return records


def print_aggregate(records: list[dict]):
    import statistics

    def avg(lst): return round(statistics.mean([x for x in lst if x is not None]), 4)

    tiers = ["easy", "medium", "hard", "all"]
    metrics = ["format", "api", "trajectory"]

    print(f"\n{'='*72}")
    print(f"{'Tier':<8} {'Metric':<12} {'Conventional':>14} {'Obfuscated':>12} {'Δ (conv-obf)':>14}")
    print(f"{'-'*72}")

    for tier in tiers:
        subset = records if tier == "all" else [r for r in records if r["tier"] == tier]
        if not subset:
            continue
        for m in metrics:
            conv_vals = [r[f"conv_{m}"] for r in subset]
            obf_vals  = [r[f"obf_{m}"]  for r in subset]
            c = avg(conv_vals)
            o = avg(obf_vals)
            d = round(c - o, 4)
            print(f"{tier:<8} {m:<12} {c:>14.4f} {o:>12.4f} {d:>+14.4f}")
        print()
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--result_dir", type=str,
                        default="precomputed_results/L4_sentiment")
    parser.add_argument("--n_cases",    type=int, default=None,
                        help="Limit to first N test cases (for quick testing)")
    parser.add_argument("--tag",        type=str, default="",
                        help="Output file suffix, e.g. 'tok20' → l4_drone_eval_tok20.json")
    args = parser.parse_args()

    run_evaluation(
        gpu=args.gpu,
        result_dir=BASE / args.result_dir,
        n_cases=args.n_cases,
        tag=args.tag,
    )
