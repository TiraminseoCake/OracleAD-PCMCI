import numpy as np


def standardize_train_test(train: np.ndarray, test: np.ndarray):
    train = train.astype(np.float32)
    test = test.astype(np.float32)

    train = np.where(np.isfinite(train), train, np.nan)
    test = np.where(np.isfinite(test), test, np.nan)

    col_mean = np.nanmean(train, axis=0, keepdims=True).astype(np.float32)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)

    train = np.where(np.isnan(train), col_mean, train).astype(np.float32)
    test = np.where(np.isnan(test), col_mean, test).astype(np.float32)

    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    var = ((train - mu) ** 2).mean(axis=0, keepdims=True).astype(np.float32)
    sd = np.sqrt(var).astype(np.float32)
    sd = np.where(sd == 0.0, 1.0, sd).astype(np.float32)

    train_z = (train - mu) / sd
    test_z = (test - mu) / sd
    return train_z.astype(np.float32), test_z.astype(np.float32), mu, sd


def reduce_label(y, T):
    y = np.asarray(y)
    if y.ndim == 2:
        y = (y.sum(axis=1) > 0).astype(np.int32)
    else:
        y = y.astype(np.int32)
    if len(y) != T:
        raise ValueError(f"label length mismatch: {len(y)} != {T}")
    return y


def anomaly_segments(y01: np.ndarray):
    y01 = np.asarray(y01).astype(np.int32)
    segs = []
    in_seg = False
    s = 0
    for i, v in enumerate(y01):
        if v == 1 and not in_seg:
            s = i
            in_seg = True
        elif v == 0 and in_seg:
            segs.append((s, i - 1))
            in_seg = False
    if in_seg:
        segs.append((s, len(y01) - 1))
    return segs


def get_median_anomaly_length(y01: np.ndarray):
    segs = anomaly_segments(y01)
    if len(segs) == 0:
        return 100
    lens = [e - s + 1 for s, e in segs]
    med = int(np.median(lens))
    return max(med, 1)


def make_pseudo_env_ids(num_windows: int, num_envs: int):
    num_envs = max(int(num_envs), 1)
    if num_envs == 1 or num_windows <= 1:
        return np.zeros((num_windows,), dtype=np.int64)
    idx = np.arange(num_windows, dtype=np.int64)
    env = np.floor(idx * num_envs / max(num_windows, 1)).astype(np.int64)
    env = np.clip(env, 0, num_envs - 1)
    return env
