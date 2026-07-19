#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export TARGET_N="${TARGET_N:-200}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
export MODEL_TAG="${MODEL_TAG:-qwen3_32b}"
export GPUS="${GPUS:-0 1}"
export RECOVERY_GPU="${RECOVERY_GPU:-0}"
export RUN_MATCHED_EVAL="${RUN_MATCHED_EVAL:-1}"
export RESULT_PREFIX="${RESULT_PREFIX:-sixcat${TARGET_N}}"
export RUNNER_PREFIX="${RUNNER_PREFIX:-6cat${TARGET_N}}"
export SELECTION_DIR="${SELECTION_DIR:-results/six_category_${TARGET_N}_selection}"

./run_6category150_full_chain.sh
