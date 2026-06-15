#!/bin/bash
# Combined CTSAD+PCMCI - PSM 5-seed (tuned: b512 lr1e-4)
CUDA_VISIBLE_DEVICES=0 python -u model/oraclead_combined_ctsad_pcmci.py \
  --input_dir data/PSM --entities PSM --dataset PSM \
  --seeds 0,1,2,3,4 --epochs 20 --batch 512 --lr 1e-4 \
  --L 10 --d 64 --heads 4 --max_lag 5 \
  --num_gat_layers 2 --gat_heads 4 --gat_dim 64 \
  --grad_clip 1.0 --prior pcmci --pcmci_alpha 0.05 --pcmci_subsample 10000 \
  --use_median_vus_window --diagnose_components --no_calibrate_scores \
  --out_dir runs/combined_psm
