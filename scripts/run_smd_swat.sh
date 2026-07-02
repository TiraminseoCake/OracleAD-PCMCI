#!/usr/bin/env bash
# Resume run after fixing the CUDA_VISIBLE_DEVICES override bug in main.py.
# PSM already completed under results/parallel/psm_20260701-221416/, so this
# script only re-runs SMD (28 entities × 4 seeds) then SWaT (1 × 4).
#
# Usage:
#   bash scripts/run_smd_swat.sh                             # foreground (tmux)
#   nohup bash scripts/run_smd_swat.sh > run_smd_swat.log 2>&1 &  # detached

set -u
GPUS="${GPUS:-0,1,2,3}"
SEEDS="${SEEDS:-0,1,2,3}"
EPOCHS="${EPOCHS:-80}"
DATE="$(date +%Y%m%d-%H%M%S)"

cd "$(dirname "$0")/.."
mkdir -p logs

run_one() {
    local ds="$1"
    local out="results/parallel/${ds,,}_${DATE}"
    local log="logs/parallel_${ds,,}_${DATE}.log"
    echo "=============================================="
    echo "[$(date +'%F %T')] START ${ds}  -> ${out}"
    echo "  log: ${log}"
    echo "=============================================="
    python scripts/run_parallel.py \
        --dataset "${ds}" \
        --gpus "${GPUS}" \
        --seeds "${SEEDS}" \
        --epochs "${EPOCHS}" \
        --out_dir "${out}" \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo "[$(date +'%F %T')] END   ${ds}  exit=${rc}"
    return "${rc}"
}

run_one SMD  || echo "[warn] SMD finished with non-zero exit"
run_one SWaT || echo "[warn] SWaT finished with non-zero exit"

echo
echo "[$(date +'%F %T')] ALL DONE."
echo "Result roots:"
ls -d results/parallel/*_"${DATE}" 2>/dev/null
