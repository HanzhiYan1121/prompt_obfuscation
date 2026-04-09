#!/bin/bash
# Wait for current batch to finish, then re-run failed/missing prompts

cd "/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

echo "=== Waiting for current jobs to complete at $(date) ==="

# Wait for all 4 current running jobs to produce best_candidate.pt
wait_for() {
    local name="$1"
    while [ ! -f "precomputed_results/$name/best_candidate.pt" ]; do
        echo "  Waiting for $name..."
        sleep 60
    done
    echo "  $name DONE"
}

wait_for L3_sci_report
wait_for L3_socratic
wait_for L3_clinical
wait_for L3_military

echo "=== All current jobs done. Starting re-run at $(date) ==="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" launch_all_experiments.py --gpus 1 2 3 4 5 2>&1 | tee logs/rerun_batch2.log

echo "=== Re-run complete at $(date) ==="
