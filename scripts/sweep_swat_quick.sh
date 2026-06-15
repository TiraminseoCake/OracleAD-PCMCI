#!/usr/bin/env bash
set -euo pipefail

RUNNER="/home/mschae/oraclead_transfer/oraclead-repro/src/runners/oraclead_npz_runner_3d_mask.py"
INPUT_DIR="/home/mschae/oraclead_transfer/processed/SWaT"
RUN_BASE="/home/mschae/oraclead_transfer/runs/swat_hparam_search"
TB_BASE="/home/mschae/oraclead_transfer/runs/tensorboard/swat_hparam_search"

mkdir -p "${RUN_BASE}" "${TB_BASE}"

COMMON_ARGS=(
  --input_dir "${INPUT_DIR}"
  --dataset SWaT

  --L 10
  --tau_max 5

  --batch 1024
  --epochs 20

  --d 64
  --heads 4
  --enc_layers 2
  --dec_layers 2

  --lam_recon 0.1
  --lam_dev 3.0

  --start_sls_epoch 5
  --weight_decay 0.01
  --seeds 0

  --pred_temp 1.0
  --lag_source_topk 0

  --p_agg topk
  --p_topk 5
  --d_agg topkrow
  --d_topk 5
  --lag_agg mean
  --lag_fusion mean

  --use_tensorboard
)

LAG_WINS=(3 5 7)
LAM_SPARSES=(0.01 0.03 0.05)
SELF_LOOP_BIASES=(0.0 0.5 1.0)

GPUS=(0 1)

run_one () {
  local gpu="$1"
  local lag_win="$2"
  local lam_sparse="$3"
  local self_loop_bias="$4"

  local exp_name="lw${lag_win}_ls${lam_sparse}_sb${self_loop_bias}"
  local out_dir="${RUN_BASE}/${exp_name}"
  local tb_dir="${TB_BASE}/${exp_name}"
  local log_file="${RUN_BASE}/${exp_name}.log"

  mkdir -p "${out_dir}" "${tb_dir}"

  echo "[RUN] ${exp_name} on GPU ${gpu}"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${RUNNER}" \
    "${COMMON_ARGS[@]}" \
    --lag_win "${lag_win}" \
    --lam_sparse "${lam_sparse}" \
    --self_loop_bias "${self_loop_bias}" \
    --tb_root "${tb_dir}" \
    --out_dir "${out_dir}" \
    > "${log_file}" 2>&1 &
}

running=0
gpu_idx=0
num_gpu=${#GPUS[@]}

for lag_win in "${LAG_WINS[@]}"; do
  for lam_sparse in "${LAM_SPARSES[@]}"; do
    for self_loop_bias in "${SELF_LOOP_BIASES[@]}"; do

      gpu="${GPUS[$gpu_idx]}"
      run_one "${gpu}" "${lag_win}" "${lam_sparse}" "${self_loop_bias}"

      running=$((running + 1))
      gpu_idx=$(((gpu_idx + 1) % num_gpu))

      if [ "${running}" -ge "${num_gpu}" ]; then
        wait -n
        running=$((running - 1))
      fi

    done
  done
done

wait
echo "[DONE] SWaT quick hyperparameter search finished."