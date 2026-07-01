"""Entity data loader for PICAAD (NPZ-backed multivariate time series).

The NPZ layout expected under `cfg.DATA.INPUT_DIR`:
    {entity}.npz  with keys:
        train  : [T_train, N]
        test   : [T_test, N]
        label  : [T_test]  or  [T_test, K]  (K anomaly types, or-reduced)
"""
import glob
import os
from typing import List

import numpy as np

from datasets.sliding_window import SlidingWindowDataset
from datasets.util import (
    make_pseudo_env_ids,
    reduce_label,
    standardize_train_test,
)


class EntityArrays:
    """Container for a single entity's standardized arrays + label + stats."""

    def __init__(self, name: str, train_z, test_z, y, mu, sd):
        self.name = name
        self.train_z = train_z
        self.test_z = test_z
        self.y = y
        self.mu = mu
        self.sd = sd
        self.N = train_z.shape[1]
        self.T_train = train_z.shape[0]
        self.T_test = test_z.shape[0]


def list_entities(cfg) -> List[str]:
    """List entity NPZ basenames under cfg.DATA.INPUT_DIR."""
    if cfg.DATA.ENTITIES:
        return [e.strip() for e in cfg.DATA.ENTITIES.split(',') if e.strip()]
    files = sorted(glob.glob(os.path.join(cfg.DATA.INPUT_DIR, '*.npz')))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_entity(cfg, entity_name: str) -> EntityArrays:
    """Read one entity NPZ file and return standardized arrays."""
    path = os.path.join(cfg.DATA.INPUT_DIR, f'{entity_name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Entity NPZ not found: {path}')

    data = np.load(path)
    train = data['train'].astype(np.float32)
    test = data['test'].astype(np.float32)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f'{entity_name}: N mismatch between train ({train.shape[1]}) '
            f'and test ({test.shape[1]})'
        )

    y = reduce_label(data['label'], test.shape[0])

    if cfg.DATA.SCALE == 'standard':
        train_z, test_z, mu, sd = standardize_train_test(train, test)
    elif cfg.DATA.SCALE == 'none':
        train_z = train.astype(np.float32)
        test_z = test.astype(np.float32)
        mu = np.zeros((1, train.shape[1]), dtype=np.float32)
        sd = np.ones((1, train.shape[1]), dtype=np.float32)
    else:
        raise ValueError(f'Unknown DATA.SCALE: {cfg.DATA.SCALE}')

    return EntityArrays(entity_name, train_z, test_z, y, mu, sd)


def build_train_dataset(cfg, train_TN) -> SlidingWindowDataset:
    L = cfg.PICAAD.L
    W = train_TN.shape[0] - L + 1
    env_ids = make_pseudo_env_ids(W, cfg.DATA.NUM_ENVS)
    return SlidingWindowDataset(train_TN, L, env_ids=env_ids, return_env=True)


def build_test_dataset(cfg, test_TN) -> SlidingWindowDataset:
    return SlidingWindowDataset(test_TN, cfg.PICAAD.L)
