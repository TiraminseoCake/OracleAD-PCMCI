import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.sliding_window import SlidingWindowDataset
from layers.ops import normalize_causal_tensor_torch
from utils.misc import robust_loc_scale, robust_zscore


def fit_score_calibrator(train_scores: dict):
    out = {}
    for key in ["P_raw", "C_raw", "G_raw"]:
        center, scale = robust_loc_scale(train_scores[key])
        out[key] = {"center": center, "scale": scale}
    return out


def apply_score_calibrator(raw_scores: dict, calibrator: dict, clip_min: float = 0.0,
                           alpha: float = 1.0, beta: float = 1.0):
    Pn = robust_zscore(raw_scores["P_raw"], calibrator["P_raw"]["center"], calibrator["P_raw"]["scale"], clip_min)
    Cn = robust_zscore(raw_scores["C_raw"], calibrator["C_raw"]["center"], calibrator["C_raw"]["scale"], clip_min)
    Gn = robust_zscore(raw_scores["G_raw"], calibrator["G_raw"]["center"], calibrator["G_raw"]["scale"], clip_min)
    S = (float(alpha) * Cn + float(beta) * Gn).astype(np.float32)
    A = (Pn * S).astype(np.float32)
    return {"P": Pn, "C": Cn, "G": Gn, "S": S, "A": A}


def score_components_to_timeline(comp_dict, Tt, start):
    out = {}
    for k, v in comp_dict.items():
        arr = np.full((Tt,), np.nan, dtype=np.float32)
        arr[start:] = np.asarray(v, dtype=np.float32)
        out[k + "_t"] = arr
    return out


def prediction_score(err: torch.Tensor, agg: str = "mean", topk: int = 3):
    if agg == "mean":
        return err.mean(dim=1)
    if agg == "max":
        return err.max(dim=1).values
    k = min(int(topk), err.shape[1])
    return err.topk(k, dim=1).values.mean(dim=1)


def matrix_deviation_per_tau(diff: torch.Tensor, agg: str = "fro", topk: int = 3):
    if agg == "fro":
        return torch.sqrt(diff.pow(2).mean(dim=(2, 3)) + 1e-12)
    row_dev = diff.abs().mean(dim=3)
    if agg == "maxrow":
        return row_dev.max(dim=2).values
    k = min(int(topk), row_dev.shape[2])
    return row_dev.topk(k, dim=2).values.mean(dim=2)


def lag_aggregate(per_tau: torch.Tensor, mode: str = "mean"):
    if mode == "max":
        return per_tau.max(dim=1).values
    return per_tau.mean(dim=1)


@torch.no_grad()
def score_windows_raw(model, series_TN, device, batch, scoring_cfg):
    """scoring_cfg: cfg.PICAAD.SCORING subtree with attributes
        P_AGG, P_TOPK, C_AGG, C_TOPK, G_AGG, G_TOPK,
        CAUSAL_LAG_AGG, GRAPH_LAG_AGG.
    """
    model.eval()
    ds = SlidingWindowDataset(series_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    W = len(ds)
    P_w = np.zeros((W,), dtype=np.float32)
    C_w = np.zeros((W,), dtype=np.float32)
    G_w = np.zeros((W,), dtype=np.float32)

    cls_ref = model.cls_ref.detach()
    w_ref = model.w_ref.detach()
    offset = 0

    for X in loader:
        X = X.to(device)
        recon, pred, C_all, pred_weights, edge_value, edge_effect, edge_strength, gate, local_delta = model(X)

        x_true_next = X[:, -1, :]
        err = (x_true_next - pred).abs()
        P = prediction_score(err, agg=scoring_cfg.P_AGG, topk=scoring_cfg.P_TOPK)

        if model.has_cls_ref:
            cls_cur = normalize_causal_tensor_torch(edge_strength)
            cdiff = cls_cur - cls_ref.unsqueeze(0)
            C_per_tau = matrix_deviation_per_tau(cdiff, agg=scoring_cfg.C_AGG, topk=scoring_cfg.C_TOPK)
            Cscore = lag_aggregate(C_per_tau, mode=scoring_cfg.CAUSAL_LAG_AGG)
        else:
            Cscore = torch.zeros_like(P)

        if pred_weights.dim() == 3:
            pred_weights_b = pred_weights.unsqueeze(0).expand(X.shape[0], -1, -1, -1)
        else:
            pred_weights_b = pred_weights

        if model.has_w_ref:
            gdiff = pred_weights_b - w_ref.unsqueeze(0)
            G_per_tau = matrix_deviation_per_tau(gdiff, agg=scoring_cfg.G_AGG, topk=scoring_cfg.G_TOPK)
            Gscore = lag_aggregate(G_per_tau, mode=scoring_cfg.GRAPH_LAG_AGG)
        else:
            Gscore = torch.zeros_like(P)

        bsz = X.shape[0]
        P_w[offset:offset + bsz] = P.detach().cpu().numpy().astype(np.float32)
        C_w[offset:offset + bsz] = Cscore.detach().cpu().numpy().astype(np.float32)
        G_w[offset:offset + bsz] = Gscore.detach().cpu().numpy().astype(np.float32)
        offset += bsz

    return {"P_raw": P_w, "C_raw": C_w, "G_raw": G_w}


def score_windows(model, series_TN, device, batch, scoring_cfg, calibrator=None):
    raw = score_windows_raw(model, series_TN, device, batch, scoring_cfg)
    if calibrator is not None:
        cal = apply_score_calibrator(
            raw,
            calibrator,
            clip_min=scoring_cfg.CALIB_CLIP_MIN,
            alpha=scoring_cfg.SCORE_ALPHA,
            beta=scoring_cfg.SCORE_BETA,
        )
    else:
        cal = {
            "P": raw["P_raw"].astype(np.float32),
            "C": raw["C_raw"].astype(np.float32),
            "G": raw["G_raw"].astype(np.float32),
        }
        cal["S"] = (scoring_cfg.SCORE_ALPHA * cal["C"] + scoring_cfg.SCORE_BETA * cal["G"]).astype(np.float32)
        cal["A"] = (cal["P"] * cal["S"]).astype(np.float32)

    out = {}
    out.update(raw)
    out.update(cal)
    return out
