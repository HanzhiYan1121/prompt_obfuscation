#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHON="${PYTHON:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/scratch/Hanzhi/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/Hanzhi/huggingface/transformers}"
export PYTHONUNBUFFERED=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen14b}"
STRUCTURED_KEYS="role,task,format,constraints,context,tool_protocol"
N_EXAMPLES="${N_EXAMPLES:-6}"
RECOVERY_BEHAVIOR_CACHE_SIZE="${RECOVERY_BEHAVIOR_CACHE_SIZE:-$N_EXAMPLES}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
CATEGORY_PROBES="${CATEGORY_PROBES:-0}"
RECOVERY_PROBE_SET="${RECOVERY_PROBE_SET:-category}"
CATEGORIES="${CATEGORIES:-role_persona task_instruction format_spec constraints context_knowledge tool_protocol}"
SELECTION_DIR="${SELECTION_DIR:-results/six_category_100_selection}"
RESULT_PREFIX="${RESULT_PREFIX:-sixcat100}"

run_category() {
  local slug="$1"
  local ids_path="${SELECTION_DIR}/${slug}_prompt_ids.json"
  local recovery_name="${RESULT_PREFIX}_${slug}_recovery${RUN_SUFFIX}"
  local structured_dir="results/${RESULT_PREFIX}_${slug}_structured_recovery${RUN_SUFFIX}"
  local lora_dir="results/${RESULT_PREFIX}_${slug}_structured_${MODEL_TAG}_lora${RUN_SUFFIX}"
  local result_glob="allprev_utility_allprev_*_tok10,prompts417_*_obf_*_tok10,prompts335_*_obf_*_tok10"
  local behavior_tag=""
  local probe_category=""

  if [ "$CATEGORY_PROBES" = "1" ]; then
    if [ "$RECOVERY_PROBE_SET" = "category" ]; then
      behavior_tag="_category_probes"
    else
      behavior_tag="_${RECOVERY_PROBE_SET}_${slug}_probes"
    fi
    probe_category="$slug"
  fi

  echo "=== [$slug] Phase 1: build recovery inputs at $(date -Is) ==="
  RESULT_GLOB="$result_glob" \
  RESULT_IDS_PATH="$ids_path" \
  RECOVERY_OUT_NAME="$recovery_name" \
  RECOVERY_N_EXAMPLES="$N_EXAMPLES" \
  RECOVERY_BEHAVIOR_CACHE_SIZE="$RECOVERY_BEHAVIOR_CACHE_SIZE" \
  RECOVERY_BEHAVIOR_OUTPUT_TAG="$behavior_tag" \
  RECOVERY_PROBE_CATEGORY="$probe_category" \
  RECOVERY_PROBE_SET="$RECOVERY_PROBE_SET" \
  RECOVERY_MAX_NEW_TOKENS=80 \
  DEDUPE_RESULT_IDS=1 \
  "$PYTHON" -u recovery_tools/build_five_prompt_recovery_inputs.py

  echo "=== [$slug] Phase 2: build structured 6D targets at $(date -Is) ==="
  "$PYTHON" -u recovery_tools/prepare_structured_6d_recovery_targets.py \
    --input "results/${recovery_name}/recovery_inputs.json" \
    --output "${structured_dir}/recovery_inputs.json"

  local total
  local train_size
  total=$("$PYTHON" - <<PY
import json
print(len(json.load(open("${structured_dir}/recovery_inputs.json", encoding="utf-8"))))
PY
)
  train_size=$("$PYTHON" - <<PY
n = int("$total")
print(max(1, min(n - 1, int(n * 0.8))))
PY
)
  echo "=== [$slug] records=$total train_size=$train_size test_size=$((total - train_size)) ==="

  echo "=== [$slug] Phase 3: train structured QLoRA at $(date -Is) ==="
  "$PYTHON" -u recovery_tools/train_acl100_recovery_lora.py \
    --gpu 0 \
    --model_id "$MODEL_ID" \
    --inputs_path "${structured_dir}/recovery_inputs.json" \
    --output_dir "$lora_dir" \
    --train_size "$train_size" \
    --epochs 5 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_length 2048 \
    --target_field target_structured_6d_json \
    --n_examples "$N_EXAMPLES" \
    --structured_json \
    --structured_keys "$STRUCTURED_KEYS"

  echo "=== [$slug] Phase 4: evaluate structured QLoRA at $(date -Is) ==="
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

  echo "=== [$slug] complete at $(date -Is) ==="
}

echo "=== Six-category Qwen recovery chain started at $(date -Is) ==="
echo "MODEL_ID=$MODEL_ID MODEL_TAG=$MODEL_TAG CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES N_EXAMPLES=$N_EXAMPLES RECOVERY_BEHAVIOR_CACHE_SIZE=$RECOVERY_BEHAVIOR_CACHE_SIZE RUN_SUFFIX=$RUN_SUFFIX CATEGORY_PROBES=$CATEGORY_PROBES RECOVERY_PROBE_SET=$RECOVERY_PROBE_SET SELECTION_DIR=$SELECTION_DIR RESULT_PREFIX=$RESULT_PREFIX CATEGORIES=$CATEGORIES"

for slug in $CATEGORIES; do
  run_category "$slug"
done

echo "=== Six-category Qwen recovery chain complete at $(date -Is) ==="
