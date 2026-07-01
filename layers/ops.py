import math

import torch


def pairwise_sq_l2(C):
    A2 = (C * C).sum(dim=2)
    G = torch.bmm(C, C.transpose(1, 2))
    D = A2.unsqueeze(2) + A2.unsqueeze(1) - 2.0 * G
    return torch.clamp(D, min=0.0)


def pairwise_l2(C):
    return (pairwise_sq_l2(C) + 1e-12).sqrt()


def logit_from_prob(p: float):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def normalize_vector_torch(x: torch.Tensor, eps: float = 1e-12):
    x = x.clamp_min(0.0)
    s = x.sum()
    if torch.isfinite(s) and float(s.detach().cpu()) > eps:
        return x / (s + eps)
    return torch.full_like(x, 1.0 / float(max(x.numel(), 1)))


def make_self_causal_fallback_torch(tau_max: int, N: int, device, dtype):
    out = torch.zeros(tau_max, N, N, device=device, dtype=dtype)
    diag = torch.arange(N, device=device)
    out[0, diag, diag] = 1.0
    return out


def normalize_causal_tensor_torch(x: torch.Tensor, eps: float = 1e-12):
    """
    Normalize nonnegative causal tensor over (tau, source) for each target.
    Supports [tau, src, tgt] and [B, tau, src, tgt].
    """
    if x.dim() == 3:
        tau_max, N, _ = x.shape
        flat = x.clamp_min(0.0).reshape(tau_max * N, N)
        colsum = flat.sum(dim=0, keepdim=True)

        fallback = torch.zeros_like(flat)
        diag = torch.arange(N, device=flat.device)
        fallback[diag, diag] = 1.0

        flat = torch.where(colsum > eps, flat / colsum.clamp_min(eps), fallback)
        return flat.view(tau_max, N, N)

    if x.dim() == 4:
        B, tau_max, N, _ = x.shape
        flat = x.clamp_min(0.0).reshape(B, tau_max * N, N)
        colsum = flat.sum(dim=1, keepdim=True)

        fallback = torch.zeros_like(flat)
        diag = torch.arange(N, device=flat.device)
        fallback[:, diag, diag] = 1.0

        flat = torch.where(colsum > eps, flat / colsum.clamp_min(eps), fallback)
        return flat.view(B, tau_max, N, N)

    raise ValueError(f"normalize_causal_tensor_torch expects 3D or 4D tensor, got {x.dim()}D.")
