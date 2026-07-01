import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Slice a [T, N] time series into length-L sliding windows for training.

    Each sample is a window x[idx:idx+L] with shape [L, N]. When `env_ids` is
    provided (used for pseudo-environment invariance loss), the sample becomes
    a tuple (x, env_id).
    """

    def __init__(self, series_TN: np.ndarray, L: int, env_ids=None, return_env: bool = False):
        self.x = series_TN.astype(np.float32)
        self.L = int(L)
        self.T, self.N = self.x.shape
        self.return_env = bool(return_env)
        if self.T < self.L:
            raise ValueError(f"T={self.T} < L={self.L}")
        self.W = self.T - self.L + 1
        if env_ids is None:
            self.env_ids = np.zeros((self.W,), dtype=np.int64)
        else:
            env_ids = np.asarray(env_ids, dtype=np.int64)
            if len(env_ids) != self.W:
                raise ValueError(f"env_ids length mismatch: {len(env_ids)} != {self.W}")
            self.env_ids = env_ids

    def __len__(self):
        return self.W

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx:idx + self.L])
        if self.return_env:
            return x, int(self.env_ids[idx])
        return x
