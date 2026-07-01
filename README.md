# CASAD: Causal Prior-Guided Anomaly Detection

## Overview

This repository implements causal prior-guided anomaly detection models for multivariate time series (MTSAD). Two main models are provided:

1. **v2\_pcmci** — PCMCI+ prior + MHSA + edge-wise prediction + multi-channel scoring
2. **Combined (CTSAD+PCMCI)** — CTSAD temporal causal GAT backbone + PCMCI+ prior + edge-wise prediction + multi-channel scoring

---

## Models

### v2\_pcmci (`model/oraclead_npz_runner_causal_v2_pcmci.py`)

Per-Variable LSTM encoder with temporal attention pooling. Each lag τ is independently encoded and passed through `nn.MultiheadAttention` (N variable nodes). PCMCI+ prior initializes learnable causal masks and edge gates.

```
Input → Per-Var LSTM + AttnPool → MHSA (N nodes, per lag) → Edge-wise Prediction → Scoring
```

**Key features:**
- PCMCI+ causal prior → edge gate / pred\_logits / attn mask initialization
- Edge-wise prediction: `pred = Σ(weight × value)` — decomposes causal contributions
- Multi-channel scoring: `A = P × (C + G)` — prediction error × structural deviation
- PCMCI+ guided intervention training

### Combined (`model/oraclead_combined_ctsad_pcmci.py`)

Uses CTSAD's temporal causal GAT as backbone. N×T nodes (variable × timestep) with directed causal edge mask. PCMCI+ prior is injected as lag-aware attention bias into the GAT layers.

```
Input → Per-Var LSTM (full seq) → Temporal Causal GAT (N×T nodes) → Edge-wise Prediction → Scoring
```

**What comes from CTSAD:**
- Per-Variable LSTM encoder (full sequence h\_seq output)
- Temporal Causal GAT with `build_causal_edge_mask` (directed temporal edges)
- N×T node graph structure

**What comes from v2\_pcmci:**
- PCMCI+ prior → `build_lag_aware_prior_bias()` → [N×T, N×T] attention bias (novel)
- Edge-wise prediction: `pred = Σ(weight × value)` (replaces CTSAD's LSTM decoder)
- Multi-channel scoring: `A = P × (C + G)` (replaces CTSAD's P × D)
- PCMCI+ guided intervention training
- Edge gate caching (fixes double-backward in intervention training)

### Other models (archived)

- `model/oraclead_npz_runner_causal_v2_gnn.py` — v2 + CausalGNNLayer (rejected: GNN hurts AD)
- `model/oraclead_npz_runner_causal_v3.py` — Shared Conv encoder + GNN (rejected: weaker C/G scores)

---

## Results

### Combined CTSAD+PCMCI (5-seed)

| Dataset | F1 | R-F1 | Aff-F | A-ROC | A-PR | V-ROC | V-PR |
|---------|-----|------|-------|-------|------|-------|------|
| **SWaT** | 78.03 | 23.07 | 81.79 | 89.05 | 78.35 | 91.58 | 79.84 |
| **PSM** (tuned) | — | — | — | — | 59.25 | — | 55.23 |
| **SMD** | 48.98 | 40.02 | 85.33 | 83.60 | 44.81 | 85.34 | 41.77 |

### v2\_pcmci (1-seed)

| Dataset | F1 | R-F1 | Aff-F | A-ROC | A-PR | V-ROC | V-PR |
|---------|-----|------|-------|-------|------|-------|------|
| **SWaT** | 77.71 | 31.61 | 76.81 | 88.40 | 79.33 | 88.25 | 73.48 |
| **PSM** (tuned) | 56.19 | 40.03 | 75.53 | 76.19 | 58.99 | 70.62 | 54.18 |

### vs OracleAD (paper, 5-seed)

| Dataset | OracleAD A-PR | Combined A-PR | Δ |
|---------|:---:|:---:|:---:|
| SWaT | 72.39 | **78.35** | +5.96 |
| PSM | **68.11** | 59.25 | -8.86 |
| SMD | **44.83** | 44.81 | -0.02 |

---

## Scripts

### Combined model experiments
| Script | Description |
|--------|-------------|
| `scripts/run_combined_swat.sh` | SWaT 5-seed (b128, lr=5e-4) |
| `scripts/run_combined_psm.sh` | PSM 5-seed (b512, lr=1e-4, tuned) |
| `scripts/run_combined_smd.sh` | SMD 5-seed (b512, lr=5e-4) |

### v2\_pcmci experiments
| Script | Description |
|--------|-------------|
| `scripts/run_swat_causal_v2.sh` | SWaT 5-seed, 3-GPU parallel |
| `scripts/run_psm_causal_v2.sh` | PSM 5-seed |
| `scripts/run_psm_causal_v2_tuned.sh` | PSM tuned (b2048, lr=1e-4) |
| `scripts/run_smd_causal_v2.sh` | SMD 5-seed |
| `scripts/run_msl_causal_v2.sh` | MSL 5-seed |
| `scripts/run_psm_extra_seeds.sh` | PSM extra seeds (5-14) |
| `scripts/run_psm_smd_parallel.sh` | PSM + SMD parallel execution |

### Hyperparameter sweeps
| Script | Description |
|--------|-------------|
| `scripts/sweep_psm_quick.sh` | PSM sweep (lr, batch, lam\_causal, te\_blend) |
| `scripts/sweep_smd_quick.sh` | SMD sweep |
| `scripts/sweep_swat_quick.sh` | SWaT sweep |

---

## Result CSVs

```
results/
├── combined/
│   ├── combined_swat_5seed.csv       # SWaT 5-seed final
│   ├── combined_psm_5seed.csv        # PSM 5-seed (default config)
│   ├── combined_smd_5seed.csv        # SMD 5-seed final
│   ├── combined_swat_1seed.csv       # SWaT 1-seed
│   ├── combined_psm_1seed.csv        # PSM 1-seed
│   ├── combined_smd_1seed.csv        # SMD 1-seed
│   ├── combined_psm_b512_lr1e4.csv   # PSM sweep best (A-PR=59.25)
│   ├── combined_psm_b1024_lr1e4.csv  # PSM sweep
│   ├── combined_psm_b2048_lr1e4.csv  # PSM sweep
│   └── combined_psm_b2048_lr5e4.csv  # PSM sweep
└── v2_pcmci/
    ├── swat_v2_pcmci.csv
    ├── swat_final_pcmci_ep.csv
    ├── swat_full_seed0.csv
    ├── psm_tuned_seed0.csv
    ├── smd_tuned.csv
    └── smd_2nd.csv
```

---

## Quick Start

### Combined model
```bash
pip install torch numpy scikit-learn pandas tigramite

python model/oraclead_combined_ctsad_pcmci.py \
  --input_dir /path/to/SWaT \
  --entities swat --dataset SWaT \
  --seeds 0,1,2,3,4 --epochs 20 --batch 128 --lr 5e-4 \
  --L 10 --d 64 --heads 4 --max_lag 5 \
  --num_gat_layers 2 --gat_heads 4 --gat_dim 64 \
  --grad_clip 1.0 --prior pcmci --pcmci_alpha 0.05 \
  --use_median_vus_window --diagnose_components
```

### v2\_pcmci
```bash
python model/oraclead_npz_runner_causal_v2_pcmci.py \
  --input_dir /path/to/SWaT \
  --entities swat --dataset SWaT \
  --epochs 80 --batch 128 --lr 5e-4 \
  --grad_clip 1.0 --prior pcmci \
  --use_median_vus_window --diagnose_components
```
