#!/usr/bin/env bash
set -u

PROJECT_DIR="/scratch/Hanzhi/paper reproduce/prompt_obfuscation"
OUTPUT_DIR="${OUTPUT_DIR:-results/inferdpt_llama31_e2e_20}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/Hanzhi/paper reproduce/venv/bin/python3}"

cd "$PROJECT_DIR" || exit 1
"$PYTHON_BIN" inferdpt_e2e/run_inferdpt_e2e.py --stage status --output_dir "$OUTPUT_DIR"
tmux ls 2>/dev/null | grep 'inferdpt_' || true
