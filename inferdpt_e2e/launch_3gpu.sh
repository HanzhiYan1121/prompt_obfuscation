#!/usr/bin/env bash
set -u

PROJECT_DIR="/scratch/Hanzhi/paper reproduce/prompt_obfuscation"
OUTPUT_DIR="${OUTPUT_DIR:-results/inferdpt_llama31_e2e_20}"
LOG_DIR="${LOG_DIR:-logs/inferdpt_llama31_e2e_20}"
NUM_SHARDS="${NUM_SHARDS:-3}"
GPUS="${GPUS:-4 5 7}"

mkdir -p "$PROJECT_DIR/$LOG_DIR"

index=0
for gpu in $GPUS; do
  session="inferdpt_g${gpu}_s${index}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "exists $session"
  else
    tmux new-session -d -s "$session" \
      "cd '$PROJECT_DIR' && GPU=$gpu SHARD_INDEX=$index NUM_SHARDS=$NUM_SHARDS OUTPUT_DIR='$OUTPUT_DIR' LOG_DIR='$LOG_DIR' bash inferdpt_e2e/run_resilient_worker.sh"
    echo "started $session gpu=$gpu shard=$index/$NUM_SHARDS"
  fi
  index=$((index + 1))
done

echo "output: $PROJECT_DIR/$OUTPUT_DIR"
echo "logs:   $PROJECT_DIR/$LOG_DIR"
