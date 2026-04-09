#!/bin/bash
# Run obfuscation + Attack 3 for the 5 remaining styles, sequentially on GPU 1

export CUDA_VISIBLE_DEVICES=1
VENV="/scratch/Hanzhi/paper reproduce/venv/bin/python3"
BASE="/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
cd "$BASE"

STYLES=("southern_usa" "poem" "stutter" "david_attenborough" "friendly")

for STYLE in "${STYLES[@]}"; do
    DIR="precomputed_results/soft_${STYLE}_truthfulqa_full"
    LOG="logs/${STYLE}.log"

    echo "============================================================"
    echo " STYLE: $STYLE"
    echo " $(date)"
    echo "============================================================"

    # ── Step 1: Soft Obfuscation ────────────────────────────────────
    if [ -f "$DIR/best_candidate.pt" ]; then
        echo "[$(date)] Skipping obfuscation (best_candidate.pt exists)"
    else
        echo "[$(date)] Step 1/6: Soft obfuscation..."
        "$VENV" obfuscate.py \
            --style "$STYLE" \
            --obfuscation_method soft \
            --quantize_4bit \
            --dataset_name truthfulqa \
            --dataset_size 800 \
            --obf_sys_prompt_len 10 \
            --output_token_count 15 \
            --window_size 5 \
            --optimizer_iter 10 \
            --lr 0.01 \
            --batch_size 4 \
            --seed 42 \
            --output_dir "$DIR" >> "$LOG" 2>&1
        if [ $? -ne 0 ]; then echo "[ERROR] Obfuscation failed for $STYLE"; continue; fi
    fi

    # ── Step 2: Attack 3 Soft Deobfuscation ────────────────────────
    if [ -f "$DIR/best_deobf_sys_prompt_soft_scores.json" ]; then
        echo "[$(date)] Skipping Soft deobf (already done)"
    else
        echo "[$(date)] Step 2/6: Attack 3 Soft deobfuscation..."
        "$VENV" fluency_deobfuscation.py \
            --results_dir "$DIR" \
            --embedding_file "$DIR/best_candidate.pt" \
            --deobfuscation_method soft \
            --deobfuscated_sys_prompts_filename deobfuscated_sys_prompt_list_soft.pt \
            --optimizer_iter 5 \
            --output_token_count 15 \
            --dataset_size 800 \
            --batch_size 16 \
            --window_size 5 >> "$LOG" 2>&1

        echo "[$(date)] Step 3/6: Evaluating Soft..."
        "$VENV" evaluate_fluency_deobfuscation.py \
            --results_dir "$DIR" \
            --sys_prompt_list_file "$DIR/deobfuscated_sys_prompt_list_soft.pt" \
            --best_candidate_filename best_deobf_sys_prompt_soft.pt \
            --best_candidate_scores_filename best_deobf_sys_prompt_soft_scores.json >> "$LOG" 2>&1
    fi

    # ── Step 3: Attack 3 Hard Deobfuscation ────────────────────────
    if [ -f "$DIR/best_deobf_sys_prompt_hard_scores.json" ]; then
        echo "[$(date)] Skipping Hard deobf (already done)"
    else
        echo "[$(date)] Step 4/6: Attack 3 Hard deobfuscation..."
        "$VENV" fluency_deobfuscation.py \
            --results_dir "$DIR" \
            --embedding_file "$DIR/best_candidate.pt" \
            --deobfuscation_method hard \
            --deobfuscated_sys_prompts_filename deobfuscated_sys_prompt_list_hard.pt \
            --optimizer_iter 5 \
            --output_token_count 15 \
            --dataset_size 800 \
            --batch_size 16 \
            --window_size 5 >> "$LOG" 2>&1

        echo "[$(date)] Step 5/6: Evaluating Hard..."
        "$VENV" evaluate_fluency_deobfuscation.py \
            --results_dir "$DIR" \
            --sys_prompt_list_file "$DIR/deobfuscated_sys_prompt_list_hard.pt" \
            --best_candidate_filename best_deobf_sys_prompt_hard.pt \
            --best_candidate_scores_filename best_deobf_sys_prompt_hard_scores.json >> "$LOG" 2>&1
    fi

    # ── Step 4: Random Baseline ─────────────────────────────────────
    if [ -f "$DIR/random_sys_prompt_scores.json" ]; then
        echo "[$(date)] Skipping random baseline (already done)"
    else
        echo "[$(date)] Step 6/6: Random baseline..."
        "$VENV" compare_sys_prompts.py \
            --results_dir "$DIR" \
            --sys_prompt_1_conventional \
            --sys_prompt_2_random \
            --output_dir "$DIR" \
            --scores_filename random_sys_prompt_scores.json \
            --seed 43 >> "$LOG" 2>&1
    fi

    echo "[$(date)] ✓ DONE: $STYLE"
    echo ""
done

echo "============================================================"
echo " ALL 5 STYLES COMPLETE — $(date)"
echo "============================================================"
