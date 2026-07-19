#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHON="${PYTHON:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"
export GPUS="${GPUS:-1 2}"
export TOKENS="${TOKENS:-10}"
export DATASET_SIZE="${DATASET_SIZE:-80}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export OPT_ITERS="${OPT_ITERS:-3}"
export RUN_EVAL="${RUN_EVAL:-0}"
export FALLBACK_LAST_ON_EVAL_FAIL="${FALLBACK_LAST_ON_EVAL_FAIL:-1}"
export PYTHONUNBUFFERED=1

CATEGORIES="${CATEGORIES:-role_persona task_instruction format_spec constraints context_knowledge tool_protocol}"
RUNNER_PREFIX="${RUNNER_PREFIX:-6cat100}"
OUT_PREFIX_TAG="${OUT_PREFIX_TAG:-$RUNNER_PREFIX}"

run_chunk() {
  local source="$1"
  local slug="$2"
  local file="system prompt/${source}_${RUNNER_PREFIX}_${slug}_runner.json"
  local n
  n=$("$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$file")
print(len(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else 0)
PY
)
  if [[ "$n" == "0" ]]; then
    echo "=== [$source/$slug] no missing obfuscations, skip ==="
    return
  fi
  export PROMPTS_FILE="$file"
  export OUT_PREFIX="${source}_${OUT_PREFIX_TAG}_${slug}_obf"
  echo "=== [$source/$slug] start missing obfuscation n=$n at $(date -Is), GPUS=$GPUS ==="
  "$PYTHON" -u run_five_prompt_obfuscation.py
  echo "=== [$source/$slug] done at $(date -Is) ==="
}

echo "=== Six-category-100 missing obfuscation started at $(date -Is) ==="
echo "Using GPUS=$GPUS CATEGORIES=$CATEGORIES RUNNER_PREFIX=$RUNNER_PREFIX OUT_PREFIX_TAG=$OUT_PREFIX_TAG"

for slug in $CATEGORIES; do
  run_chunk prompts417 "$slug"
  run_chunk prompts335 "$slug"
done

echo "=== Six-category-100 missing obfuscation complete at $(date -Is) ==="
