#!/usr/bin/env bash
# Full Step-2: clean query-disjoint black-box recovery attack evaluation.
# Evidence probes = conventional_output queries [0:6] (already used to generate
# the cached obfuscated_output evidence and the existing recovered prompts).
# Evaluation probes = queries [6:16] (query_offset=6, n_queries=10), DISJOINT
# from evidence -> out-of-distribution (query-level) attack evaluation.
# Victim model = Llama-3.1-8B-Instruct 4-bit (the architecture the tok10 soft
# prompt is bound to). Recovered prompts already exist; we only run the victim
# model on the recovered prompt over the disjoint probes and compare to the
# cached original-prompt outputs on the SAME disjoint probes.
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
QUERY_OFFSET="${QUERY_OFFSET:-6}"
N_QUERIES="${N_QUERIES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
OUTPUT_NAME="${OUTPUT_NAME:-recovery_utility_eval_disjoint_n10.json}"

LOG_DIR="logs/step2_disjoint_probe_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Step-2 disjoint-probe black-box attack eval started at $(date -Is) ==="
echo "GPU=$GPU RESULT_PREFIX=$RESULT_PREFIX MODEL_TAG=$MODEL_TAG QUERY_OFFSET=$QUERY_OFFSET N_QUERIES=$N_QUERIES"
echo "Log dir: $LOG_DIR"

for cat in $CATEGORIES; do
  lora_dir="results/${RESULT_PREFIX}_${cat}_structured_${MODEL_TAG}_lora"
  inputs_path="results/${RESULT_PREFIX}_${cat}_structured_recovery/recovery_inputs.json"

  if [ ! -d "$lora_dir" ]; then echo "[$cat] SKIP missing lora_dir: $lora_dir"; continue; fi
  if [ ! -f "$inputs_path" ]; then echo "[$cat] SKIP missing inputs_path: $inputs_path"; continue; fi

  echo "=== [$cat] disjoint probes offset=$QUERY_OFFSET n=$N_QUERIES started at $(date -Is) ==="
  "$PYTHON" -u recovery_tools/eval_recovery_utility.py \
    --lora_dir "$lora_dir" \
    --inputs_path "$inputs_path" \
    --gpu 0 \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --n_queries "$N_QUERIES" \
    --query_offset "$QUERY_OFFSET" \
    --conventional_output_name conventional_output.json \
    --output_name "$OUTPUT_NAME" \
    --structured_keys "$STRUCTURED_KEYS" \
    2>&1 | tee "$LOG_DIR/${cat}_disjoint_eval.log"
  echo "=== [$cat] complete at $(date -Is) ==="
done

echo "=== Step-2 disjoint-probe black-box attack eval complete at $(date -Is) ==="
