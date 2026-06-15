#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2.py"
GPU=3

mkdir -p logs runs/smd_causal_v2

# SMD: 28 entities, all N=38, Ttr~24K-29K
# First pass: 1 seed, 80 epochs per entity to see overall trends
# lr=5e-4 (PSM tuning showed 5e-4 is better than 5e-5 for non-SWaT)

COMMON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/SMD
  --dataset OTHER
  --epochs 80
  --batch 512
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
  --diagnose_components
  --no_cte
  --seeds 0
  --out_dir runs/smd_causal_v2
)

echo "[$(date)] Starting SMD 28 entities (1 seed each)"

CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
  "${COMMON_ARGS[@]}" \
  > "logs/smd_causal_v2.log" 2>&1

echo "[$(date)] SMD done."
