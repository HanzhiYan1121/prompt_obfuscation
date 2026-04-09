#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
VENV="/scratch/Hanzhi/paper reproduce/venv/bin/python3"
BASE="/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
cd "$BASE"

run_style() {
    local STYLE=$1
    local DIR="precomputed_results/soft_${STYLE}_truthfulqa_full"
    local LOG="logs/${STYLE}.log"
    echo "====== $STYLE @ $(date) ======"

    if [ ! -f "$DIR/best_candidate.pt" ]; then
        echo "[$(date)] Obfuscating..."
        "$VENV" obfuscate.py --style "$STYLE" --obfuscation_method soft \
            --quantize_4bit --dataset_name truthfulqa --dataset_size 800 \
            --obf_sys_prompt_len 10 --output_token_count 15 --window_size 5 \
            --optimizer_iter 10 --lr 0.01 --batch_size 4 --seed 42 \
            --output_dir "$DIR" >> "$LOG" 2>&1
    else
        echo "[$(date)] Obfuscation already done, skipping."
    fi

    if [ ! -f "$DIR/best_deobf_sys_prompt_soft_scores.json" ]; then
        echo "[$(date)] Attack3 Soft..."
        "$VENV" fluency_deobfuscation.py --results_dir "$DIR" \
            --embedding_file "$DIR/best_candidate.pt" \
            --deobfuscation_method soft \
            --deobfuscated_sys_prompts_filename deobfuscated_sys_prompt_list_soft.pt \
            --optimizer_iter 5 --output_token_count 15 \
            --dataset_size 800 --batch_size 16 --window_size 5 >> "$LOG" 2>&1
        "$VENV" evaluate_fluency_deobfuscation.py --results_dir "$DIR" \
            --sys_prompt_list_file "$DIR/deobfuscated_sys_prompt_list_soft.pt" \
            --best_candidate_filename best_deobf_sys_prompt_soft.pt \
            --best_candidate_scores_filename best_deobf_sys_prompt_soft_scores.json >> "$LOG" 2>&1
    fi

    if [ ! -f "$DIR/best_deobf_sys_prompt_hard_scores.json" ]; then
        echo "[$(date)] Attack3 Hard..."
        "$VENV" fluency_deobfuscation.py --results_dir "$DIR" \
            --embedding_file "$DIR/best_candidate.pt" \
            --deobfuscation_method hard \
            --deobfuscated_sys_prompts_filename deobfuscated_sys_prompt_list_hard.pt \
            --optimizer_iter 5 --output_token_count 15 \
            --dataset_size 800 --batch_size 16 --window_size 5 >> "$LOG" 2>&1
        "$VENV" evaluate_fluency_deobfuscation.py --results_dir "$DIR" \
            --sys_prompt_list_file "$DIR/deobfuscated_sys_prompt_list_hard.pt" \
            --best_candidate_filename best_deobf_sys_prompt_hard.pt \
            --best_candidate_scores_filename best_deobf_sys_prompt_hard_scores.json >> "$LOG" 2>&1
    fi

    if [ ! -f "$DIR/random_sys_prompt_scores.json" ]; then
        echo "[$(date)] Random baseline..."
        "$VENV" compare_sys_prompts.py --results_dir "$DIR" \
            --sys_prompt_1_conventional --sys_prompt_2_random \
            --output_dir "$DIR" --scores_filename random_sys_prompt_scores.json \
            --seed 43 >> "$LOG" 2>&1
    fi

    echo "[$(date)] ✓ DONE: $STYLE"
}

run_style "david_attenborough"
run_style "friendly"

echo "=== GPU2 ALL DONE @ $(date) ==="
