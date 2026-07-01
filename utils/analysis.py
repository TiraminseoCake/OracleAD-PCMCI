"""Post-hoc intervention contribution analysis for PICAAD.

Runs per-window interventions on a trained model to measure how much each
(tau, source) edge contributes to prediction error, then reports/persists
top-k edges. Used for interpretability and visualization; not needed for
training or scoring.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.sliding_window import SlidingWindowDataset
from model.intervention import intervene_local_window


@torch.no_grad()
def compute_intervention_contribution_3d(model, test_TN, device, batch,
                                          mode: str = "permute", fill_value: float = 0.0):
    model.eval()
    ds = SlidingWindowDataset(test_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    tau_max = model.tau_max
    N = model.N

    raw_sum = np.zeros((tau_max, N, N), dtype=np.float64)
    pos_sum = np.zeros((tau_max, N, N), dtype=np.float64)
    n_windows = 0

    for X in loader:
        X = X.to(device)
        x_true_next = X[:, -1, :]
        _, pred_base, _, _, _, _, _, _, _ = model(X)
        base_err = (x_true_next - pred_base).abs()
        B = X.shape[0]

        for tau in range(1, tau_max + 1):
            for src in range(N):
                if mode == "fill":
                    _, pred_int, _, _, _, _, _, _, _ = model(
                        X, mask_tau=tau, mask_var=src, mask_fill_value=fill_value,
                    )
                else:
                    Xp = intervene_local_window(
                        X, tau=tau, src=src, lag_win=model.lag_win,
                        mode=mode, fill_value=fill_value,
                    )
                    _, pred_int, _, _, _, _, _, _, _ = model(Xp)

                int_err = (x_true_next - pred_int).abs()
                delta = int_err - base_err
                raw_sum[tau - 1, src, :] += delta.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                pos_sum[tau - 1, src, :] += torch.clamp(delta, min=0.0).sum(dim=0).detach().cpu().numpy().astype(np.float64)
        n_windows += B

    if n_windows == 0:
        raise RuntimeError("No test windows available for intervention contribution analysis.")

    G_raw_tau = raw_sum / float(n_windows)
    G_pos_tau = pos_sum / float(n_windows)
    return {
        "G_raw_tau":           G_raw_tau.astype(np.float32),
        "G_pos_tau":           G_pos_tau.astype(np.float32),
        "G_raw_lag_mean":      G_raw_tau.mean(axis=0).astype(np.float32),
        "G_pos_lag_mean":      G_pos_tau.mean(axis=0).astype(np.float32),
        "G_raw_lag_max":       G_raw_tau.max(axis=0).astype(np.float32),
        "G_pos_lag_max":       G_pos_tau.max(axis=0).astype(np.float32),
        "source_strength_tau": G_pos_tau.sum(axis=2).astype(np.float32),
        "target_received_tau": G_pos_tau.sum(axis=1).astype(np.float32),
    }


def topk_edges_from_matrix(M: np.ndarray, topk: int):
    M = np.asarray(M)
    N1, N2 = M.shape
    flat = M.reshape(-1)
    order = np.argsort(flat)[::-1]
    out = []
    for idx in order:
        val = flat[idx]
        if not np.isfinite(val):
            continue
        src = idx // N2
        tgt = idx % N2
        out.append((src, tgt, float(val)))
        if len(out) >= topk:
            break
    return out


def topk_edges_from_tensor(T: np.ndarray, topk: int):
    T = np.asarray(T)
    tau_max, N1, N2 = T.shape
    flat = T.reshape(-1)
    order = np.argsort(flat)[::-1]
    out = []
    for idx in order:
        val = flat[idx]
        if not np.isfinite(val):
            continue
        tau = idx // (N1 * N2)
        rem = idx % (N1 * N2)
        src = rem // N2
        tgt = rem % N2
        out.append((tau + 1, src, tgt, float(val)))
        if len(out) >= topk:
            break
    return out


def print_intervention_contrib_summary(name: str, G_pos_tau: np.ndarray, topk: int = 10):
    G_pos_lag_mean = G_pos_tau.mean(axis=0)
    print(f"\n[{name}] intervention contribution top-{topk} edges (lag-mean, positive delta)", flush=True)
    for rank, (src, tgt, val) in enumerate(topk_edges_from_matrix(G_pos_lag_mean, topk), start=1):
        print(f"  {rank:02d}. src={src:02d} -> tgt={tgt:02d} : {val:.6f}", flush=True)
    print(f"[{name}] intervention contribution top-{topk} lag-specific edges", flush=True)
    for rank, (tau, src, tgt, val) in enumerate(topk_edges_from_tensor(G_pos_tau, topk), start=1):
        print(f"  {rank:02d}. tau={tau:02d} src={src:02d} -> tgt={tgt:02d} : {val:.6f}", flush=True)


def save_intervention_contrib_csv(csv_path: str, G_raw_tau: np.ndarray, G_pos_tau: np.ndarray):
    import pandas as pd
    tau_max, N, _ = G_raw_tau.shape
    rows = []
    for tau in range(tau_max):
        for src in range(N):
            for tgt in range(N):
                rows.append({
                    "tau": tau + 1, "source": src, "target": tgt,
                    "raw_delta":      float(G_raw_tau[tau, src, tgt]),
                    "positive_delta": float(G_pos_tau[tau, src, tgt]),
                })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
