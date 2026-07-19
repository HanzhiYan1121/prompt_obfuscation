#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHON="${PYTHON:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
export HF_HOME="${HF_HOME:-/scratch/Hanzhi/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/Hanzhi/huggingface/transformers}"
export PYTHONUNBUFFERED=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
MODEL_TAG="${MODEL_TAG:-qwen3_32b}"
RESULT_PREFIX="${RESULT_PREFIX:-privacybench100}"
METHODS="${METHODS:-confusionprompt inferdpt split_denoise confidential_prompting noir}"
CATEGORIES="${CATEGORIES:-role_persona task_instruction format_spec constraints context_knowledge tool_protocol}"
STRUCTURED_KEYS="${STRUCTURED_KEYS:-role,task,format,constraints,context,tool_protocol}"
N_EXAMPLES="${N_EXAMPLES:-6}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
UTILITY_MODEL="${UTILITY_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
UTILITY_QUERIES="${UTILITY_QUERIES:-6}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPUS=\"0 1 ...\"." >&2
  exit 1
fi

mkdir -p logs/privacybench100_5defense

run_one() {
  local method="$1"
  local category="$2"
  local gpu="$3"
  local structured_dir="results/${RESULT_PREFIX}_${method}_${category}_structured_recovery"
  local lora_dir="results/${RESULT_PREFIX}_${method}_${category}_structured_${MODEL_TAG}_lora"
  local log="logs/privacybench100_5defense/${method}_${category}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "=== [$method/$category] start $(date -Is) gpu=$gpu ==="

    if [ ! -f "${structured_dir}/recovery_inputs.json" ]; then
      echo "Missing structured inputs: ${structured_dir}/recovery_inputs.json" >&2
      exit 1
    fi

    if [ ! -d "${lora_dir}/adapter" ]; then
      echo "=== [$method/$category] train QLoRA $(date -Is) ==="
      "$PYTHON" -u recovery_tools/train_acl100_recovery_lora.py \
        --gpu 0 \
        --model_id "$MODEL_ID" \
        --inputs_path "${structured_dir}/recovery_inputs.json" \
        --output_dir "$lora_dir" \
        --train_size 80 \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --lora_r 16 \
        --lora_alpha 32 \
        --max_length "$MAX_LENGTH" \
        --target_field target_structured_6d_json \
        --n_examples "$N_EXAMPLES" \
        --structured_json \
        --structured_keys "$STRUCTURED_KEYS"
    else
      echo "=== [$method/$category] skip train, adapter exists ==="
    fi

    if [ ! -f "${lora_dir}/lora_recovered_prompts.json" ]; then
      echo "=== [$method/$category] recover held-out prompts $(date -Is) ==="
      "$PYTHON" -u recovery_tools/eval_acl100_recovery_lora.py \
        --gpu 0 \
        --model_id "$MODEL_ID" \
        --inputs_path "${structured_dir}/recovery_inputs.json" \
        --lora_dir "$lora_dir" \
        --max_new_tokens 180 \
        --target_field target_structured_6d_json \
        --n_examples "$N_EXAMPLES" \
        --structured_json \
        --structured_keys "$STRUCTURED_KEYS"
    else
      echo "=== [$method/$category] skip recovery, lora_recovered_prompts.json exists ==="
    fi

    if [ ! -f "${lora_dir}/recovery_utility_eval_matched_n6.json" ]; then
      echo "=== [$method/$category] utility eval $(date -Is) ==="
      "$PYTHON" -u recovery_tools/eval_recovery_utility.py \
        --lora_dir "$lora_dir" \
        --inputs_path "${structured_dir}/recovery_inputs.json" \
        --gpu 0 \
        --model_name "$UTILITY_MODEL" \
        --eval_batch_size 4 \
        --n_queries "$UTILITY_QUERIES" \
        --output_name recovery_utility_eval_matched_n6.json \
        --structured_keys "$STRUCTURED_KEYS"
    else
      echo "=== [$method/$category] skip utility eval, output exists ==="
    fi

    echo "=== [$method/$category] complete $(date -Is) ==="
  ) > "$log" 2>&1
}

echo "=== privacybench100 5-defense recovery start $(date -Is) ==="
echo "MODEL_ID=$MODEL_ID MODEL_TAG=$MODEL_TAG METHODS=$METHODS CATEGORIES=$CATEGORIES"
echo "GPUS=${GPU_LIST[*]} parallel_jobs=${#GPU_LIST[@]}"

for method in $METHODS; do
  echo "=== method wave: $method $(date -Is) ==="
  pids=()
  idx=0
  for category in $CATEGORIES; do
    gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
    run_one "$method" "$category" "$gpu" &
    pids+=("$!")
    idx=$((idx + 1))

    if [ "${#pids[@]}" -ge "${#GPU_LIST[@]}" ]; then
      fail=0
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
          fail=1
        fi
      done
      if [ "$fail" != "0" ]; then
        echo "Method wave failed: $method" >&2
        exit 1
      fi
      pids=()
    fi
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [ "$fail" != "0" ]; then
    echo "Method wave failed: $method" >&2
    exit 1
  fi
  echo "=== method wave complete: $method $(date -Is) ==="
done

echo "=== privacybench100 5-defense recovery complete $(date -Is) ==="
