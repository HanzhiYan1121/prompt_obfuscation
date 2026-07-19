#!/usr/bin/env bash
# Full-coverage Step-2: generate evidence-disjoint original-prompt outputs for
# ALL 40 held-out prompts per category, then run the query-disjoint utility eval
# over all of them (no cached-query subset bias).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHON="${PYTHON:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
export HF_HOME="${HF_HOME:-/scratch/Hanzhi/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/Hanzhi/huggingface/transformers}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

GPU="${GPU:-4}"
export CUDA_VISIBLE_DEVICES="$GPU"

STRUCTURED_KEYS="role,task,format,constraints,context,tool_protocol"
CATEGORIES="${CATEGORIES:-role_persona task_instruction format_spec constraints context_knowledge tool_protocol}"
RESULT_PREFIX="${RESULT_PREFIX:-sixcat200}"
MODEL_TAG="${MODEL_TAG:-qwen3_32b}"
N_QUERIES="${N_QUERIES:-10}"
OFFSET="${OFFSET:-6}"
CONV_NAME="conventional_output_disjoint.json"
OUTPUT_NAME="recovery_utility_eval_disjoint_full40.json"

LOG_DIR="logs/step2_disjoint_full40_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Step-2 full-40 disjoint attack started at $(date -Is) ==="
echo "GPU=$GPU OFFSET=$OFFSET N_QUERIES=$N_QUERIES  log=$LOG_DIR"

echo "=== Stage A: generate evidence-disjoint original-prompt outputs (all held-out) ==="
"$PYTHON" -u recovery_tools/gen_disjoint_conventional_outputs.py \
  --result_prefix "$RESULT_PREFIX" \
  --model_tag "$MODEL_TAG" \
  --categories "$CATEGORIES" \
  --offset "$OFFSET" \
  --n_queries "$N_QUERIES" \
  --output_name "$CONV_NAME" \
  --overwrite \
  2>&1 | tee "$LOG_DIR/gen_disjoint_outputs.log"

echo "=== Stage B: query-disjoint utility eval over all held-out prompts ==="
for cat in $CATEGORIES; do
  lora_dir="results/${RESULT_PREFIX}_${cat}_structured_${MODEL_TAG}_lora"
  inputs_path="results/${RESULT_PREFIX}_${cat}_structured_recovery/recovery_inputs.json"
  [ -d "$lora_dir" ] || { echo "[$cat] SKIP missing $lora_dir"; continue; }

  echo "=== [$cat] full-40 disjoint eval started at $(date -Is) ==="
  "$PYTHON" -u recovery_tools/eval_recovery_utility.py \
    --lora_dir "$lora_dir" \
    --inputs_path "$inputs_path" \
    --gpu 0 \
    --eval_batch_size 8 \
    --n_queries "$N_QUERIES" \
    --query_offset 0 \
    --conventional_output_name "$CONV_NAME" \
    --output_name "$OUTPUT_NAME" \
    --structured_keys "$STRUCTURED_KEYS" \
    2>&1 | tee "$LOG_DIR/${cat}_full40_eval.log"
  echo "=== [$cat] complete at $(date -Is) ==="
done

echo "=== Step-2 full-40 disjoint attack complete at $(date -Is) ==="
