#!/bin/bash
# Run multi-turn utility degradation experiment.
# Tests whether the 10-token soft prompt maintains utility across multi-turn conversations.
#
# Usage:
#   bash bash_scripts/run_multi_turn_eval.sh [results_dir] [gpu_id]
#
# Defaults to pirate style on GPU 0.

RESULTS_DIR=${1:-"precomputed_results/soft_pirate_truthfulqa_full"}
GPU=${2:-0}
VENV="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

echo "=== Multi-Turn Degradation Experiment ==="
echo "Results dir : $RESULTS_DIR"
echo "GPU         : $GPU"
echo ""

CUDA_VISIBLE_DEVICES=$GPU $VENV eval_multi_turn.py \
    --results_dir "$RESULTS_DIR" \
    --n_triplets 30 \
    --batch_size 4 \
    --output_dir results \
    --output_name multi_turn_degradation.json \
    2>&1 | tee logs/multi_turn_eval.log

echo ""
echo "Done. Results saved to results/multi_turn_degradation.json"
