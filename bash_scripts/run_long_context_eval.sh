#!/bin/bash
# Run long-context utility degradation experiment.
# Tests whether the 10-token soft prompt maintains utility as user messages grow longer.
#
# Usage:
#   bash bash_scripts/run_long_context_eval.sh [results_dir] [gpu_id]
#
# Defaults to pirate style on GPU 0.

RESULTS_DIR=${1:-"precomputed_results/soft_pirate_truthfulqa_full"}
GPU=${2:-0}
VENV="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

echo "=== Long Context Degradation Experiment ==="
echo "Results dir : $RESULTS_DIR"
echo "GPU         : $GPU"
echo ""

CUDA_VISIBLE_DEVICES=$GPU $VENV eval_long_context.py \
    --results_dir "$RESULTS_DIR" \
    --context_lengths 0 128 256 512 1024 \
    --n_samples 40 \
    --batch_size 4 \
    --output_dir results \
    --output_name long_context_degradation.json \
    2>&1 | tee logs/long_context_eval.log

echo ""
echo "Done. Results saved to results/long_context_degradation.json"
