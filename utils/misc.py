import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_devices(visible_devices):
    if visible_devices is None or visible_devices == '':
        return
    if isinstance(visible_devices, (list, tuple)):
        val = ','.join(str(x) for x in visible_devices)
    else:
        val = str(visible_devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = val


def mkdir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pct(x):
    return (float(x) * 100.0) if np.isfinite(x) else float("nan")


def safe_mean_std(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def robust_loc_scale(arr, eps: float = 1e-6):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return med, scale


def robust_zscore(arr, center, scale, clip_min=0.0):
    z = (np.asarray(arr, dtype=np.float64) - float(center)) / max(float(scale), 1e-6)
    if clip_min is not None:
        z = np.maximum(z, float(clip_min))
    return z.astype(np.float32)
