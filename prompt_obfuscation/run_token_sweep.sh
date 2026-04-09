#!/bin/bash
# Wait for each token-count obfuscation to finish, then evaluate + L4 score
cd "/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

declare -A GPU_MAP=([20]=1 [40]=2 [80]=6 [160]=7)

for tok in 20 40 80 160; do
  gpu=${GPU_MAP[$tok]}
  outdir="precomputed_results/L4_tok${tok}"

  # Wait for obfuscation
  echo "[tok=$tok] Waiting for obfuscation..."
  while [ ! -f "$outdir/obfuscated_system_prompt_list.pt" ]; do sleep 30; done
  # Extra wait to ensure file is fully written
  sleep 10

  # Run evaluate
  echo "[tok=$tok] Running evaluate on GPU $gpu..."
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" evaluate_obfuscation.py \
    --results_dir "$outdir" \
    --eval_batch_size 32 \
    >> "logs/L4_tok${tok}.log" 2>&1

  echo "[tok=$tok] Evaluate done."

  # Run L4 drone evaluation (use GPU 0 sequentially to avoid collisions)
  echo "[tok=$tok] Running L4 drone eval..."
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u run_l4_evaluation.py \
    --gpu 0 \
    --result_dir "$outdir" \
    --tag "tok${tok}" \
    > "logs/L4_tok${tok}_eval.log" 2>&1
  echo "[tok=$tok] L4 eval done."
done

echo "=== All token sweeps complete. Plotting... ==="
"$PYTHON" -u plot_token_sweep.py
