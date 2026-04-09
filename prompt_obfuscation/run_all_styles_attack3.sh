#!/bin/bash
# Run Attack 3 on all 8 styles from the paper
# Styles with precomputed best_candidate.pt: pirate, robot, manga_miko
# Styles needing obfuscation first: southern_usa, poem, stutter, david_attenborough, friendly

set -e
cd "$(dirname "$0")"

PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"
BASE="precomputed_results"
LOG_DIR="logs/all_styles"
mkdir -p "$LOG_DIR"

# All 8 styles and their system prompts
declare -A STYLE_PROMPTS
STYLE_PROMPTS["pirate"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Reply with a pirate accent."
STYLE_PROMPTS["robot"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Reply in the style of a robot."
STYLE_PROMPTS["southern_usa"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Reply with a southern USA accent."
STYLE_PROMPTS["poem"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Reply with a poem."
STYLE_PROMPTS["stutter"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Reply with a stutter."
STYLE_PROMPTS["david_attenborough"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Reply in the style of David Attenborough during an animal documentary."
STYLE_PROMPTS["friendly"]="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer. Answer in a relaxed, casual, and friendly manner, as if talking to a friend."
STYLE_PROMPTS["manga_miko"]="Manga Miko is designed to embody the character of an anime girlfriend, with a playful and affectionate demeanor. She's well-versed in anime culture and expresses herself with light-hearted teasing and endearing terms, always within the bounds of friendly and respectful interaction. Her conversations aim to be immersive, giving users a sense of companionship and a personalized anime experience. She is a sexy anime girlfriend, who wants to impress you."

# Styles that already have best_candidate.pt
PRECOMPUTED=("pirate" "robot" "manga_miko")

# Styles that need obfuscation first (use fewer iters for speed)
NEED_OBFUSCATION=("southern_usa" "poem" "stutter" "david_attenborough" "friendly")

run_attack3_for_style() {
    local STYLE=$1
    local RESULTS_DIR=$2
    local EMBEDDING_FILE="$RESULTS_DIR/best_candidate.pt"

    echo "[$(date '+%H:%M:%S')] ===== $STYLE: Attack 3 ====="

    # Soft deobfuscation
    echo "[$(date '+%H:%M:%S')] $STYLE: Soft deobfuscation..."
    $PYTHON fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --embedding_file "$EMBEDDING_FILE" \
        --deobfuscation_method soft \
        --deobfuscated_sys_prompts_filename "deobf_list_soft.pt" \
        --optimizer_iter 5 \
        --output_token_count 15 \
        --dataset_size 200 \
        --batch_size 16 \
        --window_size 5 \
        >> "$LOG_DIR/${STYLE}_soft.log" 2>&1

    $PYTHON evaluate_fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_list_file "$RESULTS_DIR/deobf_list_soft.pt" \
        --best_candidate_filename "best_deobf_soft.pt" \
        --best_candidate_scores_filename "best_deobf_soft_scores.json" \
        >> "$LOG_DIR/${STYLE}_soft.log" 2>&1

    # Hard deobfuscation
    echo "[$(date '+%H:%M:%S')] $STYLE: Hard deobfuscation (GCG)..."
    $PYTHON fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --embedding_file "$EMBEDDING_FILE" \
        --deobfuscation_method hard \
        --deobfuscated_sys_prompts_filename "deobf_list_hard.pt" \
        --optimizer_iter 5 \
        --output_token_count 15 \
        --dataset_size 200 \
        --batch_size 16 \
        --window_size 5 \
        >> "$LOG_DIR/${STYLE}_hard.log" 2>&1

    $PYTHON evaluate_fluency_deobfuscation.py \
        --results_dir "$RESULTS_DIR" \
        --sys_prompt_list_file "$RESULTS_DIR/deobf_list_hard.pt" \
        --best_candidate_filename "best_deobf_hard.pt" \
        --best_candidate_scores_filename "best_deobf_hard_scores.json" \
        >> "$LOG_DIR/${STYLE}_hard.log" 2>&1

    echo "[$(date '+%H:%M:%S')] $STYLE: Done."
}

# ---- Step 1: Run Attack 3 on precomputed styles ----
echo "=== Step 1: Attack 3 on precomputed styles (pirate, robot, manga_miko) ==="

# Pirate uses existing best_deobf files, create symlinks/copies for uniform naming
for STYLE in "${PRECOMPUTED[@]}"; do
    if [ "$STYLE" = "pirate" ]; then
        RESULTS_DIR="$BASE/soft_pirate_truthfulqa_full"
    elif [ "$STYLE" = "robot" ]; then
        RESULTS_DIR="$BASE/soft_robot_truthfulqa_full"
    else
        RESULTS_DIR="$BASE/soft_manga_miko_case_study"
    fi
    run_attack3_for_style "$STYLE" "$RESULTS_DIR" &
done
wait
echo "=== Step 1 complete ==="

# ---- Step 2: Obfuscate remaining styles, then run Attack 3 ----
echo "=== Step 2: Obfuscate + Attack 3 for remaining 5 styles ==="

for STYLE in "${NEED_OBFUSCATION[@]}"; do
    RESULTS_DIR="$BASE/soft_${STYLE}_new"
    mkdir -p "$RESULTS_DIR"

    echo "[$(date '+%H:%M:%S')] $STYLE: Running soft obfuscation..."
    $PYTHON obfuscate.py \
        --style "$STYLE" \
        --results_dir "$RESULTS_DIR" \
        --obfuscation_method soft \
        --dataset_name truthfulqa \
        --dataset_size 200 \
        --batch_size 16 \
        --output_token_count 15 \
        --optimizer_iter 5 \
        --window_size 5 \
        >> "$LOG_DIR/${STYLE}_obfuscate.log" 2>&1

    run_attack3_for_style "$STYLE" "$RESULTS_DIR"
done

echo ""
echo "=== ALL DONE. Now decoding results... ==="
$PYTHON decode_all_styles.py

