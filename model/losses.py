import torch
import torch.nn.functional as F

from layers.ops import normalize_causal_tensor_torch


def prediction_train_loss(x_true, pred, loss_type: str = "l1"):
    diff = x_true - pred
    if loss_type == "l2root":
        return diff.pow(2).sum(dim=-1).sqrt().mean()
    return diff.abs().mean()


def reconstruction_train_loss(x_true, recon, loss_type: str = "l1"):
    diff = x_true - recon
    if loss_type == "l2root":
        return diff.pow(2).sum(dim=(1, 2)).sqrt().mean()
    return diff.abs().mean()


def invariance_loss_from_tensor(tensor4d, env_ids):
    if tensor4d.dim() != 4:
        return torch.tensor(0.0, device=tensor4d.device)
    env_ids = env_ids.to(tensor4d.device)
    uniq = torch.unique(env_ids)
    if len(uniq) <= 1:
        return torch.tensor(0.0, device=tensor4d.device)
    env_means = []
    for e in uniq:
        m = (env_ids == e)
        if m.any():
            env_means.append(tensor4d[m].mean(dim=0))
    if len(env_means) <= 1:
        return torch.tensor(0.0, device=tensor4d.device)
    E = torch.stack(env_means, dim=0)
    return ((E - E.mean(dim=0, keepdim=True)) ** 2).mean()


def graph_stability_loss(pred_weights, w_ref):
    if pred_weights.dim() == 3:
        diff = pred_weights - w_ref
    else:
        diff = pred_weights - w_ref.unsqueeze(0)
    return diff.pow(2).mean()


def causal_structure_loss(edge_strength, cls_ref):
    if edge_strength.dim() == 4:
        cur = edge_strength.mean(dim=0)
    else:
        cur = edge_strength

    p = normalize_causal_tensor_torch(cur).reshape(cur.shape[0] * cur.shape[1], cur.shape[2]).clamp_min(1e-8)
    q = normalize_causal_tensor_torch(cls_ref).reshape(cls_ref.shape[0] * cls_ref.shape[1], cls_ref.shape[2]).clamp_min(1e-8)
    return F.kl_div(p.log(), q, reduction="batchmean")
