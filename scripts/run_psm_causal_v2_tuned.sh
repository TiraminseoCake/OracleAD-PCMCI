#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2.py"
GPU=3

mkdir -p logs runs/psm_causal_v2_tuned

COMMON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/PSM
  --entities PSM
  --dataset PSM
  --epochs 80
  --batch 1024
  --lr 5e-4
  --L 10
  --tau_max 5
  --lag_win 5
  --d 64
  --heads 4
  --enc_layers 2
  --dec_layers 2
  --grad_clip 1.0
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
  --no_cte
)

for seed in 0 1 2 3 4; do
  echo "[$(date)] Starting PSM tuned seed $seed"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
    "${COMMON_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/psm_causal_v2_tuned/seed${seed}" \
    > "logs/psm_causal_v2_tuned_seed${seed}.log" 2>&1
  echo "[$(date)] Done seed $seed"
done

echo "[$(date)] PSM tuned all seeds done."
