#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2.py"
GPU=3

mkdir -p logs runs/psm_causal_v2_extra

COMMON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/PSM
  --entities PSM
  --dataset PSM
  --epochs 80
  --batch 1024
  --lr 5e-4
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --use_median_vus_window
  --diagnose_components
  --no_cte
)

for seed in 5 6 7 8 9 10 11 12 13 14; do
  echo "[$(date)] Starting PSM seed $seed"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
    "${COMMON_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/psm_causal_v2_extra/seed${seed}" \
    > "logs/psm_extra_seed${seed}.log" 2>&1
  echo "[$(date)] Done seed $seed"
  grep "seed ${seed}.*A-PR" "logs/psm_extra_seed${seed}.log" 2>/dev/null
done

echo ""
echo "=== ALL PSM SEEDS SUMMARY ==="
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  printf "seed %2d: " "$seed"
  if [ "$seed" -le 4 ]; then
    grep "seed ${seed}.*A-PR" "logs/psm_causal_v2_tuned_seed${seed}.log" 2>/dev/null || echo "(missing)"
  else
    grep "seed ${seed}.*A-PR" "logs/psm_extra_seed${seed}.log" 2>/dev/null || echo "(missing)"
  fi
done
