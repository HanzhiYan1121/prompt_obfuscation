#!/usr/bin/env bash
set -u

GPU="${GPU:?set GPU, for example GPU=4}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX, for example SHARD_INDEX=0}"
NUM_SHARDS="${NUM_SHARDS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/emojiprompt_llama31_e2e_20}"
LOG_DIR="${LOG_DIR:-logs/emojiprompt_llama31_e2e_20}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/Hanzhi/huggingface/transformers}"
SEMANTIC_METRIC="${SEMANTIC_METRIC:-bertscore}"
MAX_OBFUSCATION_ATTEMPTS="${MAX_OBFUSCATION_ATTEMPTS:-10}"
MAX_ATOMIC_UNITS="${MAX_ATOMIC_UNITS:-24}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "$LOG_DIR"

attempt=0
last_code=1
while [ "$attempt" -le "$MAX_RESTARTS" ]; do
  {
    date -u
    echo "worker start gpu=$GPU shard=$SHARD_INDEX/$NUM_SHARDS attempt=$attempt"
  } >> "$LOG_DIR/worker_${SHARD_INDEX}.log"

  CUDA_VISIBLE_DEVICES="$GPU" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  "$PYTHON_BIN" emojiprompt_e2e/run_emojiprompt_e2e.py \
    --stage worker \
    --output_dir "$OUTPUT_DIR" \
    --n_prompts 20 \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$SHARD_INDEX" \
    --semantic_metric "$SEMANTIC_METRIC" \
    --max_obfuscation_attempts "$MAX_OBFUSCATION_ATTEMPTS" \
    --max_atomic_units "$MAX_ATOMIC_UNITS" \
    --batch_size "$BATCH_SIZE" \
    >> "$LOG_DIR/worker_${SHARD_INDEX}.log" 2>&1

  last_code="$?"
  if [ "$last_code" -eq 0 ]; then
    date -u >> "$LOG_DIR/worker_${SHARD_INDEX}.log"
    echo "worker complete gpu=$GPU shard=$SHARD_INDEX/$NUM_SHARDS" >> "$LOG_DIR/worker_${SHARD_INDEX}.log"
    exit 0
  fi

  attempt=$((attempt + 1))
  {
    date -u
    echo "worker failed code=$last_code gpu=$GPU shard=$SHARD_INDEX/$NUM_SHARDS; retry=$attempt/$MAX_RESTARTS"
  } >> "$LOG_DIR/worker_${SHARD_INDEX}.log"
  sleep 60
done

exit "$last_code"
