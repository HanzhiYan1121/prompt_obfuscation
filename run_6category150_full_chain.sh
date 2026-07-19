#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHON="${PYTHON:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
export HF_HOME="${HF_HOME:-/scratch/Hanzhi/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/Hanzhi/huggingface/transformers}"
export PYTHONUNBUFFERED=1

TARGET_N="${TARGET_N:-150}"
SELECTION_DIR="${SELECTION_DIR:-results/six_category_${TARGET_N}_selection}"
RUNNER_PREFIX="${RUNNER_PREFIX:-6cat${TARGET_N}}"
RESULT_PREFIX="${RESULT_PREFIX:-sixcat${TARGET_N}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen14b}"
CATEGORIES="${CATEGORIES:-role_persona task_instruction format_spec constraints context_knowledge tool_protocol}"
GPUS="${GPUS:-0 1}"
RECOVERY_GPU="${RECOVERY_GPU:-0}"
RUN_MATCHED_EVAL="${RUN_MATCHED_EVAL:-1}"

echo "=== Six-category-${TARGET_N} full chain started at $(date -Is) ==="
echo "SELECTION_DIR=$SELECTION_DIR RUNNER_PREFIX=$RUNNER_PREFIX RESULT_PREFIX=$RESULT_PREFIX MODEL_ID=$MODEL_ID MODEL_TAG=$MODEL_TAG"

echo "=== Step 1: prepare six category selections ==="
"$PYTHON" dataset_tools/prepare_6category100_selection.py \
  --target_n "$TARGET_N" \
  --seed 42 \
  --output_dir "$SELECTION_DIR" \
  --runner_prefix "$RUNNER_PREFIX"

echo "=== Step 2: run missing obfuscations ==="
GPUS="$GPUS" \
RUNNER_PREFIX="$RUNNER_PREFIX" \
OUT_PREFIX_TAG="$RUNNER_PREFIX" \
CATEGORIES="$CATEGORIES" \
./run_6category100_obfuscation.sh

echo "=== Step 3: run structured QLoRA recovery ==="
CUDA_VISIBLE_DEVICES="$RECOVERY_GPU" \
SELECTION_DIR="$SELECTION_DIR" \
RESULT_PREFIX="$RESULT_PREFIX" \
MODEL_ID="$MODEL_ID" \
MODEL_TAG="$MODEL_TAG" \
CATEGORIES="$CATEGORIES" \
./run_6category100_qwen_recovery.sh

if [ "$RUN_MATCHED_EVAL" = "1" ]; then
  echo "=== Step 4: run matched n=6 Utility F1 eval ==="
  CUDA_VISIBLE_DEVICES="$RECOVERY_GPU" \
  RESULT_PREFIX="$RESULT_PREFIX" \
  MODEL_TAG="$MODEL_TAG" \
  CATEGORIES="$CATEGORIES" \
  RUN_SUFFIX="" \
  ./run_6category100_matched_probe_eval.sh
fi

echo "=== Six-category-${TARGET_N} full chain complete at $(date -Is) ==="
