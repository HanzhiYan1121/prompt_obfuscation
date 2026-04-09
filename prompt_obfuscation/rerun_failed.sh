#!/bin/bash
# Re-run failed + never-started prompts on GPUs 1 2 3 4 5
# Run after the first batch completes

cd "/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting failed/missing re-run at $(date) ==="
"$PYTHON" launch_all_experiments.py --gpus 1 2 3 4 5 2>&1 | tee logs/rerun_batch2.log
echo "=== Done at $(date) ==="
