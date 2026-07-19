#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHON="${PYTHON:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
export HF_HOME="${HF_HOME:-/scratch/Hanzhi/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/Hanzhi/huggingface/transformers}"
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

STRUCTURED_KEYS="role,task,format,constraints,context,tool_protocol"
CATEGORIES="${CATEGORIES:-role_persona task_instruction format_spec constraints context_knowledge tool_protocol}"
RESULT_PREFIX="${RESULT_PREFIX:-sixcat100}"
MODEL_TAG="${MODEL_TAG:-qwen14b}"
RUN_SUFFIX="${RUN_SUFFIX-_catprobe}"
N_QUERIES="${N_QUERIES:-6}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
OUTPUT_NAME="${OUTPUT_NAME:-recovery_utility_eval_matched_n6.json}"

LOG_DIR="logs/sixcat100_matched_probe_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Six-category matched-probe utility eval started at $(date -Is) ==="
echo "GPU=$GPU MODEL_TAG=$MODEL_TAG RUN_SUFFIX=$RUN_SUFFIX N_QUERIES=$N_QUERIES OUTPUT_NAME=$OUTPUT_NAME"
echo "Log dir: $LOG_DIR"

for cat in $CATEGORIES; do
  lora_dir="results/${RESULT_PREFIX}_${cat}_structured_${MODEL_TAG}_lora${RUN_SUFFIX}"
  inputs_path="results/${RESULT_PREFIX}_${cat}_structured_recovery${RUN_SUFFIX}/recovery_inputs.json"

  if [ ! -d "$lora_dir" ]; then
    echo "[$cat] SKIP missing lora_dir: $lora_dir"
    continue
  fi
  if [ ! -f "$inputs_path" ]; then
    echo "[$cat] SKIP missing inputs_path: $inputs_path"
    continue
  fi

  echo "=== [$cat] matched original-test probes started at $(date -Is) ==="
  "$PYTHON" -u recovery_tools/eval_recovery_utility.py \
    --lora_dir "$lora_dir" \
    --inputs_path "$inputs_path" \
    --gpu 0 \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --n_queries "$N_QUERIES" \
    --conventional_output_name conventional_output.json \
    --output_name "$OUTPUT_NAME" \
    --structured_keys "$STRUCTURED_KEYS" \
    2>&1 | tee "$LOG_DIR/${cat}_matched_eval.log"
  echo "=== [$cat] matched original-test probes complete at $(date -Is) ==="
done

echo "=== Six-category matched-probe utility eval complete at $(date -Is) ==="
