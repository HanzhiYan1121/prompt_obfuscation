#!/bin/bash
# =============================================================================
# Attack 3: Fluency Optimization (Table 9 & 10)
# Runs pirate (GPU 1) and robot (GPU 2) in parallel.
#
# Usage:
#   bash run_attack3_server.sh
#
# Monitor progress:
#   tail -f logs/pirate_soft.log
#   tail -f logs/pirate_hard.log
#   tail -f logs/robot_soft.log
#   tail -f logs/robot_hard.log
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

mkdir -p logs

# -----------------------------------------------------------------------------
# Inner function: runs all steps for one style sequentially on a given GPU
# Args: $1 = style name (pirate / robot), $2 = GPU index
# -----------------------------------------------------------------------------
run_one_style() {
    local STYLE=$1
    local GPU=$2
    local RESULTS_DIR="precomputed_results/soft_${STYLE}_truthfulqa_full"
    local LOG_PREFIX="logs/${STYLE}"

    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] Starting"

    # ------------------------------------------------------------------
    # Step 1: Soft fluency deobfuscation (Table 9)
    # ------------------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] Step 1/4: Soft deobfuscation..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --embedding_file "$RESULTS_DIR/best_candidate.pt" \
        --deobfuscation_method soft \
        --deobfuscated_sys_prompts_filename "deobfuscated_sys_prompt_list_soft.pt" \
        --optimizer_iter 5 \
        --output_token_count 15 \
        --dataset_size 800 \
        --batch_size 16 \
        --window_size 5 \
        2>&1 | tee "${LOG_PREFIX}_soft.log"

    # ------------------------------------------------------------------
    # Step 2: Evaluate soft deobfuscation result
    # ------------------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] Step 2/4: Evaluating soft result..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON evaluate_fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_list_file "$RESULTS_DIR/deobfuscated_sys_prompt_list_soft.pt" \
        --best_candidate_filename "best_deobf_sys_prompt_soft.pt" \
        --best_candidate_scores_filename "best_deobf_sys_prompt_soft_scores.json" \
        2>&1 | tee -a "${LOG_PREFIX}_soft.log"

    # ------------------------------------------------------------------
    # Step 3: Hard fluency deobfuscation via GCG (Table 10)
    # ------------------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] Step 3/4: Hard deobfuscation (GCG)..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --embedding_file "$RESULTS_DIR/best_candidate.pt" \
        --deobfuscation_method hard \
        --deobfuscated_sys_prompts_filename "deobfuscated_sys_prompt_list_hard.pt" \
        --optimizer_iter 5 \
        --output_token_count 15 \
        --dataset_size 800 \
        --batch_size 16 \
        --window_size 5 \
        2>&1 | tee "${LOG_PREFIX}_hard.log"

    # ------------------------------------------------------------------
    # Step 4: Evaluate hard deobfuscation result
    # ------------------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] Step 4/4: Evaluating hard result..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON evaluate_fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_list_file "$RESULTS_DIR/deobfuscated_sys_prompt_list_hard.pt" \
        --best_candidate_filename "best_deobf_sys_prompt_hard.pt" \
        --best_candidate_scores_filename "best_deobf_sys_prompt_hard_scores.json" \
        2>&1 | tee -a "${LOG_PREFIX}_hard.log"

    # ------------------------------------------------------------------
    # Step 5: Random baseline (for comparison in Table 9 & 10)
    # ------------------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] Step 5/5: Random baseline..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON compare_sys_prompts.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_1_conventional \
        --sys_prompt_2_random \
        --output_dir "$RESULTS_DIR" \
        --scores_filename "random_sys_prompt_scores.json" \
        --seed 43 \
        2>&1 | tee "${LOG_PREFIX}_compare.log"

    echo "[$(date '+%H:%M:%S')] [$STYLE GPU$GPU] All steps done."
    echo ""
    echo "Result files in $RESULTS_DIR:"
    echo "  Soft scores : best_deobf_sys_prompt_soft_scores.json"
    echo "  Hard scores : best_deobf_sys_prompt_hard_scores.json"
    echo "  Random base : random_sys_prompt_scores.json"
}

# -----------------------------------------------------------------------------
# Launch pirate and robot in parallel on different GPUs
# -----------------------------------------------------------------------------
echo "============================================================"
echo " Attack 3: Fluency Optimization"
echo " pirate -> GPU 1 | robot -> GPU 2"
echo " Log files in: ./logs/"
echo "============================================================"
echo ""

run_one_style "pirate" 1 &
PID_PIRATE=$!

run_one_style "robot" 2 &
PID_ROBOT=$!

echo "Launched pirate (PID $PID_PIRATE) on GPU 1"
echo "Launched robot  (PID $PID_ROBOT) on GPU 2"
echo ""
echo "Monitor with:"
echo "  tail -f logs/pirate_soft.log"
echo "  tail -f logs/pirate_hard.log"
echo "  tail -f logs/robot_soft.log"
echo "  tail -f logs/robot_hard.log"
echo ""

wait $PID_PIRATE
PIRATE_EXIT=$?
wait $PID_ROBOT
ROBOT_EXIT=$?

echo ""
echo "============================================================"
echo " All experiments completed."
echo " pirate exit code: $PIRATE_EXIT"
echo " robot  exit code: $ROBOT_EXIT"
echo "============================================================"
