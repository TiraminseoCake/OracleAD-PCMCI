#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs runs/psm_g1 runs/smd_g2_partA runs/smd_g3_partB

COMMON_ARGS=(
  --epochs 80
  --batch 128
  --L 10
  --tau_max 5
  --lag_win 5
  --d 64
  --heads 4
  --enc_layers 2
  --dec_layers 2
  --lam_recon 0.1
  --lam_dev 3.0
  --lam_graph 1.0
  --lam_gate 0.02
  --lam_entropy 0.01
  --lam_lagmono 0.01
  --lam_perm 0.05
  --lam_inv 0.02
  --num_envs 4
  --perm_pairs_per_batch 1
  --graph_hidden 8
  --dynamic_graph
  --diagnose_components
  --use_median_vus_window
  --save_per_seed
)

readarray -t SMD_SPLIT < <(python - <<'PY'
import glob, os
files = sorted(glob.glob('/home/mschae/oraclead_transfer/processed/SMD/*.npz'))
names = [os.path.splitext(os.path.basename(f))[0] for f in files]
if not names:
    raise SystemExit("No SMD npz files found in /home/mschae/oraclead_transfer/processed/SMD")
a = names[::2]
b = names[1::2]
print(",".join(a))
print(",".join(b))
PY
)

SMD_A="${SMD_SPLIT[0]}"
SMD_B="${SMD_SPLIT[1]}"

echo "SMD split A:"
echo "$SMD_A"
echo
echo "SMD split B:"
echo "$SMD_B"
echo

CUDA_VISIBLE_DEVICES=1 python -u src/runners/oraclead_npz_runner_3d_dev.py \
  "${COMMON_ARGS[@]}" \
  --input_dir /home/mschae/oraclead_transfer/processed/PSM \
  --dataset PSM \
  --seeds 0,1,2,3,4 \
  --out_dir runs/psm_g1 \
  > logs/psm_g1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u src/runners/oraclead_npz_runner_3d_dev.py \
  "${COMMON_ARGS[@]}" \
  --input_dir /home/mschae/oraclead_transfer/processed/SMD \
  --dataset SMD \
  --entities "$SMD_A" \
  --seeds 0,1,2,3,4 \
  --out_dir runs/smd_g2_partA \
  > logs/smd_g2_partA.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u src/runners/oraclead_npz_runner_3d_dev.py \
  "${COMMON_ARGS[@]}" \
  --input_dir /home/mschae/oraclead_transfer/processed/SMD \
  --dataset SMD \
  --entities "$SMD_B" \
  --seeds 0,1,2,3,4 \
  --out_dir runs/smd_g3_partB \
  > logs/smd_g3_partB.log 2>&1 &

wait
