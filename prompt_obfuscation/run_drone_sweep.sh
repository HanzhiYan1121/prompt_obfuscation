#!/bin/bash
# Wait for drone-trained obfuscation, then evaluate + L4 score for each token count
cd "/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

declare -A GPU_MAP=([10]=4 [20]=6 [40]=2 [80]=7 [160]=0)

for tok in 10 20 40 80 160; do
  gpu=${GPU_MAP[$tok]}
  outdir="precomputed_results/drone_tok${tok}"

  echo "[drone tok=$tok] Waiting for obfuscation..."
  while [ ! -f "$outdir/obfuscated_system_prompt_list.pt" ]; do sleep 30; done
  sleep 10

  echo "[drone tok=$tok] Running evaluate on GPU $gpu..."
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" evaluate_obfuscation.py \
    --results_dir "$outdir" \
    --eval_batch_size 32 \
    >> "logs/drone_tok${tok}.log" 2>&1
  echo "[drone tok=$tok] Evaluate done."

  echo "[drone tok=$tok] Running L4 drone eval on GPU 4..."
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -u run_l4_evaluation.py \
    --gpu 0 \
    --result_dir "$outdir" \
    --tag "drone_tok${tok}" \
    > "logs/drone_tok${tok}_eval.log" 2>&1
  echo "[drone tok=$tok] L4 eval done."
done

echo "=== All drone sweeps complete. Plotting... ==="
"$PYTHON" -u plot_drone_sweep.py
