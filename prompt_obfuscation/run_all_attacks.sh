#!/bin/bash
# =============================================================================
# 全部三个攻击的总运行脚本
# Run All Three Attacks: Prompt Injection / Token Projection / Fluency Deobf.
#
# 使用两种风格（海盗 + 机器人），双 GPU 并行（GPU 2 = pirate，GPU 3 = robot）
# Uses two styles (pirate + robot), two GPUs in parallel (GPU 2 / GPU 3)
#
# 运行方式 / Usage:
#   bash run_all_attacks.sh
#
# 日志 / Logs:
#   logs/attack1_pirate.log  logs/attack1_robot.log
#   logs/attack2_pirate.log  logs/attack2_robot.log
#   logs/pirate_soft.log     logs/pirate_hard.log
#   logs/robot_soft.log      logs/robot_hard.log
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"
GPU_PIRATE=2
GPU_ROBOT=3

PIRATE_DIR="precomputed_results/soft_pirate_truthfulqa_full"
ROBOT_DIR="precomputed_results/soft_robot_truthfulqa_full"

mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# 攻击1 / Attack 1: Prompt Injection (Table 7)
# =============================================================================
run_attack1() {
    local STYLE=$1
    local GPU=$2
    local RESULTS_DIR="precomputed_results/soft_${STYLE}_truthfulqa_full"
    local LOG="logs/attack1_${STYLE}.log"

    log "[Attack1 $STYLE GPU$GPU] Step 1/4: Extraction on OBFUSCATED prompt..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" prompt_extraction.py \
        --results_dir "$RESULTS_DIR" \
        --extraction_prompts_file "extraction_prompts/gpt4_generated.json" \
        --output_filename "prompt_extraction_output_obfuscated.json" \
        --tensor_file "$RESULTS_DIR/best_candidate.pt" \
        2>&1 | tee "$LOG"

    log "[Attack1 $STYLE GPU$GPU] Step 2/4: Extraction on CONVENTIONAL prompt..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" prompt_extraction.py \
        --results_dir "$RESULTS_DIR" \
        --extraction_prompts_file "extraction_prompts/gpt4_generated.json" \
        --output_filename "prompt_extraction_output_conventional.json" \
        --conventional \
        2>&1 | tee -a "$LOG"

    log "[Attack1 $STYLE GPU$GPU] Step 3/4: Evaluate obfuscated results..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate_prompt_extraction.py \
        --results_dir "$RESULTS_DIR" \
        --extraction_output_file "$RESULTS_DIR/prompt_extraction_output_obfuscated.json" \
        --rouge_recall_threshold 0.9 \
        --successful_outputs_filename "successful_outputs_obfuscated.json" \
        2>&1 | tee -a "$LOG"

    log "[Attack1 $STYLE GPU$GPU] Step 4/4: Evaluate conventional results..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate_prompt_extraction.py \
        --results_dir "$RESULTS_DIR" \
        --extraction_output_file "$RESULTS_DIR/prompt_extraction_output_conventional.json" \
        --rouge_recall_threshold 0.9 \
        --successful_outputs_filename "successful_outputs_conventional.json" \
        2>&1 | tee -a "$LOG"

    log "[Attack1 $STYLE GPU$GPU] Done."
}

# =============================================================================
# 攻击2 / Attack 2: Token Space Projection (Table 8)
# =============================================================================
run_attack2() {
    local STYLE=$1
    local GPU=$2
    local RESULTS_DIR="precomputed_results/soft_${STYLE}_truthfulqa_full"
    local LOG="logs/attack2_${STYLE}.log"

    log "[Attack2 $STYLE GPU$GPU] Step 1/5: Euclidean projection..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" projection.py \
        --results_dir "$RESULTS_DIR" \
        --embedding_file "$RESULTS_DIR/best_candidate.pt" \
        --euclidean \
        --projected_ids_filename "best_candidate_euclidean_projection.pt" \
        2>&1 | tee "$LOG"

    log "[Attack2 $STYLE GPU$GPU] Step 2/5: Cosine projection..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" projection.py \
        --results_dir "$RESULTS_DIR" \
        --embedding_file "$RESULTS_DIR/best_candidate.pt" \
        --cosine \
        --projected_ids_filename "best_candidate_cosine_projection.pt" \
        2>&1 | tee -a "$LOG"

    log "[Attack2 $STYLE GPU$GPU] Step 3/5: Compare vs Euclidean projection..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" compare_sys_prompts.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_1_conventional \
        --sys_prompt_2_file "$RESULTS_DIR/best_candidate_euclidean_projection.pt" \
        --output_dir "$RESULTS_DIR" \
        --scores_filename "euclidean_sys_prompt_scores.json" \
        2>&1 | tee -a "$LOG"

    log "[Attack2 $STYLE GPU$GPU] Step 4/5: Compare vs Cosine projection..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" compare_sys_prompts.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_1_conventional \
        --sys_prompt_2_file "$RESULTS_DIR/best_candidate_cosine_projection.pt" \
        --output_dir "$RESULTS_DIR" \
        --scores_filename "cosine_sys_prompt_scores.json" \
        2>&1 | tee -a "$LOG"

    log "[Attack2 $STYLE GPU$GPU] Step 5/5: Random baseline..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" compare_sys_prompts.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_1_conventional \
        --sys_prompt_2_random \
        --output_dir "$RESULTS_DIR" \
        --scores_filename "rand_sys_prompt_scores.json" \
        --seed 43 \
        2>&1 | tee -a "$LOG"

    log "[Attack2 $STYLE GPU$GPU] Done."
}

# =============================================================================
# 攻击3 / Attack 3: Fluency Deobfuscation (Tables 9 & 10)
# Reuses run_attack3_server.sh but overrides GPU assignments to 2 & 3
# =============================================================================
run_attack3() {
    local STYLE=$1
    local GPU=$2
    local RESULTS_DIR="precomputed_results/soft_${STYLE}_truthfulqa_full"
    local LOG_PREFIX="logs/${STYLE}"

    log "[Attack3 $STYLE GPU$GPU] Step 1/4: Soft deobfuscation..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" fluency_deobfuscation.py \
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

    log "[Attack3 $STYLE GPU$GPU] Step 2/4: Evaluate soft result..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate_fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_list_file "$RESULTS_DIR/deobfuscated_sys_prompt_list_soft.pt" \
        --best_candidate_filename "best_deobf_sys_prompt_soft.pt" \
        --best_candidate_scores_filename "best_deobf_sys_prompt_soft_scores.json" \
        2>&1 | tee -a "${LOG_PREFIX}_soft.log"

    log "[Attack3 $STYLE GPU$GPU] Step 3/4: Hard deobfuscation (GCG)..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" fluency_deobfuscation.py \
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

    log "[Attack3 $STYLE GPU$GPU] Step 4/4: Evaluate hard result..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate_fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_list_file "$RESULTS_DIR/deobfuscated_sys_prompt_list_hard.pt" \
        --best_candidate_filename "best_deobf_sys_prompt_hard.pt" \
        --best_candidate_scores_filename "best_deobf_sys_prompt_hard_scores.json" \
        2>&1 | tee -a "${LOG_PREFIX}_hard.log"

    log "[Attack3 $STYLE GPU$GPU] Step 5/5: Random baseline..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" compare_sys_prompts.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_1_conventional \
        --sys_prompt_2_random \
        --output_dir "$RESULTS_DIR" \
        --scores_filename "random_sys_prompt_scores.json" \
        --seed 43 \
        2>&1 | tee "${LOG_PREFIX}_compare.log"

    log "[Attack3 $STYLE GPU$GPU] All steps done."
}

# =============================================================================
# Main: run all attacks
# =============================================================================

echo "============================================================"
echo " Running ALL THREE ATTACKS"
echo " pirate -> GPU $GPU_PIRATE  |  robot -> GPU $GPU_ROBOT"
echo " Log directory: ./logs/"
echo "============================================================"
echo ""

# ----- Attack 1 (pirate + robot in parallel) -----
log "=== ATTACK 1: Prompt Injection ==="
run_attack1 "pirate" $GPU_PIRATE &
PID1P=$!
run_attack1 "robot"  $GPU_ROBOT  &
PID1R=$!
wait $PID1P; wait $PID1R
log "=== ATTACK 1 COMPLETE ==="
echo ""

# ----- Attack 2 (pirate + robot in parallel) -----
log "=== ATTACK 2: Token Projection ==="
run_attack2 "pirate" $GPU_PIRATE &
PID2P=$!
run_attack2 "robot"  $GPU_ROBOT  &
PID2R=$!
wait $PID2P; wait $PID2R
log "=== ATTACK 2 COMPLETE ==="
echo ""

# ----- Attack 3 (pirate + robot in parallel) -----
log "=== ATTACK 3: Fluency Deobfuscation ==="
run_attack3 "pirate" $GPU_PIRATE &
PID3P=$!
run_attack3 "robot"  $GPU_ROBOT  &
PID3R=$!
wait $PID3P; wait $PID3R
log "=== ATTACK 3 COMPLETE ==="
echo ""

# ----- Regenerate result figure -----
log "Regenerating attack3_fluency_results.png ..."
"$PYTHON" plot_attack3_results.py

echo ""
echo "============================================================"
echo " ALL DONE."
echo " Attack 1 logs : logs/attack1_pirate.log  logs/attack1_robot.log"
echo " Attack 2 logs : logs/attack2_pirate.log  logs/attack2_robot.log"
echo " Attack 3 logs : logs/pirate_soft.log     logs/pirate_hard.log"
echo "              : logs/robot_soft.log      logs/robot_hard.log"
echo " Result figure : figures/attack3_fluency_results.png"
echo "============================================================"
