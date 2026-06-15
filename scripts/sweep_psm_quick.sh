#!/usr/bin/env bash
set -euo pipefail

# PSM quick hyperparameter sweep on GPU 3
# 1 seed, 30 epochs per config — enough to see trends
# Key variables: lr, batch, lam_causal, score combination

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2.py"
GPU=3
OUTBASE="runs/psm_sweep"
mkdir -p logs "$OUTBASE"

COMMON=(
  --input_dir /home/mschae/oraclead_transfer/processed/PSM
  --entities PSM
  --dataset PSM
  --epochs 30
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --use_median_vus_window
  --diagnose_components
  --no_cte
  --seeds 0
)

run_config() {
  local tag="$1"; shift
  echo "[$(date)] Starting $tag"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
    "${COMMON[@]}" "$@" \
    --out_dir "$OUTBASE/$tag" \
    > "logs/psm_sweep_${tag}.log" 2>&1
  echo "[$(date)] Done $tag"
  grep 'paper_eval components' -A5 "logs/psm_sweep_${tag}.log" 2>/dev/null
  grep 'seed 0.*A-PR' "logs/psm_sweep_${tag}.log" 2>/dev/null
  echo "---"
}

# Baseline (current config)
run_config "base_lr5e5_b1024" --lr 5e-5 --batch 1024

# Higher lr (PSM is smaller, may need more aggressive lr)
run_config "lr1e4_b1024" --lr 1e-4 --batch 1024
run_config "lr5e4_b1024" --lr 5e-4 --batch 1024
run_config "lr1e3_b1024" --lr 1e-3 --batch 1024

# Smaller batch (more gradient updates per epoch)
run_config "lr5e4_b256" --lr 5e-4 --batch 256
run_config "lr1e3_b256" --lr 1e-3 --batch 256

# Reduce causal loss weight (C/G are noisy on PSM, let P dominate)
run_config "lr5e4_b256_lowcausal" --lr 5e-4 --batch 256 --lam_causal 0.1 --lam_robust 0.01

# Higher TE blend (stronger prior guidance)
run_config "lr5e4_b256_teblend07" --lr 5e-4 --batch 256 --te_prior_blend 0.7

echo ""
echo "============================================"
echo "SWEEP SUMMARY"
echo "============================================"
for tag in base_lr5e5_b1024 lr1e4_b1024 lr5e4_b1024 lr1e3_b1024 lr5e4_b256 lr1e3_b256 lr5e4_b256_lowcausal lr5e4_b256_teblend07; do
  printf "%-30s " "$tag"
  grep 'seed 0.*A-PR' "logs/psm_sweep_${tag}.log" 2>/dev/null || echo "(not done)"
done
