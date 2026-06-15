#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2.py"
GPU=2
OUTBASE="runs/smd_sweep"
mkdir -p logs "$OUTBASE"

# 대표 entity 3개: 고성능(machine-1-1), 중간(machine-2-4), 저성능(machine-1-2)
ENTITIES="machine-1-1,machine-2-4,machine-1-2"

COMMON=(
  --input_dir /home/mschae/oraclead_transfer/processed/SMD
  --entities "$ENTITIES"
  --dataset OTHER
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
    > "logs/smd_sweep_${tag}.log" 2>&1
  echo "[$(date)] Done $tag"
  grep 'seed 0.*A-PR' "logs/smd_sweep_${tag}.log" 2>/dev/null
  echo "---"
}

# Baseline (current config)
run_config "base_lr5e4_b512" --lr 5e-4 --batch 512

# lr variations
run_config "lr1e3_b512" --lr 1e-3 --batch 512
run_config "lr1e4_b512" --lr 1e-4 --batch 512

# batch variations
run_config "lr5e4_b256" --lr 5e-4 --batch 256
run_config "lr1e3_b256" --lr 1e-3 --batch 256

# lower causal weight (C/G may be noisy for some entities)
run_config "lr5e4_b512_lowcausal" --lr 5e-4 --batch 512 --lam_causal 0.1 --lam_robust 0.01

# higher lr with smaller model capacity test
run_config "lr1e3_b512_d32" --lr 1e-3 --batch 512 --d 32

echo ""
echo "============================================"
echo "SMD SWEEP SUMMARY"
echo "============================================"
for tag in base_lr5e4_b512 lr1e3_b512 lr1e4_b512 lr5e4_b256 lr1e3_b256 lr5e4_b512_lowcausal lr1e3_b512_d32; do
  echo "=== $tag ==="
  grep 'seed 0.*A-PR' "logs/smd_sweep_${tag}.log" 2>/dev/null || echo "(not done)"
done
