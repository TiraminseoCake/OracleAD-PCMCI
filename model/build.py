import hashlib
import os
import time

import numpy as np
import torch

from model.modeling_picaad import PICAAD
from model.priors import (
    build_cte_causal_prior,
    build_pcmci_causal_prior,
    build_te_causal_prior,
)


def build_model(cfg, N: int) -> PICAAD:
    """Instantiate PICAAD from cfg. N (number of variables) is entity-specific
    and passed in explicitly since it isn't part of the base yacs schema.
    """
    if cfg.MODEL.NAME != 'PICAAD':
        raise ValueError(f'Unknown MODEL.NAME: {cfg.MODEL.NAME}')

    model = PICAAD(
        N=N,
        L=cfg.PICAAD.L,
        tau_max=cfg.PICAAD.TAU_MAX,
        d=cfg.PICAAD.D,
        heads=cfg.PICAAD.HEADS,
        enc_layers=cfg.PICAAD.ENC_LAYERS,
        dec_layers=cfg.PICAAD.DEC_LAYERS,
        dropout=cfg.PICAAD.DROPOUT,
        mhsa_residual=cfg.PICAAD.MHSA_RESIDUAL,
        lag_fusion=cfg.PICAAD.LAG_FUSION,
        lag_win=cfg.PICAAD.LAG_WIN,
        pred_temp=cfg.PICAAD.PRED_TEMP,
        self_loop_bias=cfg.PICAAD.SELF_LOOP_BIAS,
        lag_source_topk=cfg.PICAAD.LAG_SOURCE_TOPK,
        dynamic_graph=cfg.PICAAD.DYNAMIC_GRAPH,
        graph_hidden=cfg.PICAAD.GRAPH_HIDDEN,
        gate_init=cfg.PICAAD.GATE_INIT,
        te_prior_blend=cfg.PICAAD.PRIOR.BLEND,
        causal_attn_mask_scale=cfg.PICAAD.CAUSAL_ATTN_MASK_SCALE,
        causal_mask_warmup_epochs=cfg.PICAAD.CAUSAL_MASK_WARMUP,
    )
    return model


def build_causal_prior(cfg, train_TN: np.ndarray):
    """Build (te_weight, te_gate) NPZ-compatible arrays from cfg + train data."""
    prior_type = cfg.PICAAD.PRIOR.TYPE
    tau_max = cfg.PICAAD.TAU_MAX

    if prior_type == 'pcmci':
        return build_pcmci_causal_prior(
            train_TN,
            tau_max=tau_max,
            ci_test=cfg.PICAAD.PRIOR.PCMCI.CI_TEST,
            pc_alpha=cfg.PICAAD.PRIOR.PCMCI.ALPHA,
            subsample=cfg.PICAAD.PRIOR.PCMCI.SUBSAMPLE,
            self_mass=cfg.PICAAD.PRIOR.SELF_MASS,
            seed=cfg.PICAAD.PRIOR.SEED,
        )
    if prior_type == 'cte':
        return build_cte_causal_prior(
            train_TN,
            tau_max=tau_max,
            num_bins=cfg.PICAAD.PRIOR.TE.BINS,
            num_chunks=cfg.PICAAD.PRIOR.TE.NUM_CHUNKS,
            chunk_len=cfg.PICAAD.PRIOR.TE.CHUNK_LEN,
            threshold=cfg.PICAAD.PRIOR.TE.THRESHOLD,
            self_mass=cfg.PICAAD.PRIOR.SELF_MASS,
            seed=cfg.PICAAD.PRIOR.SEED,
        )
    if prior_type == 'te':
        return build_te_causal_prior(
            train_TN,
            tau_max=tau_max,
            num_bins=cfg.PICAAD.PRIOR.TE.BINS,
            num_chunks=cfg.PICAAD.PRIOR.TE.NUM_CHUNKS,
            chunk_len=cfg.PICAAD.PRIOR.TE.CHUNK_LEN,
            threshold=cfg.PICAAD.PRIOR.TE.THRESHOLD,
            self_mass=cfg.PICAAD.PRIOR.SELF_MASS,
            seed=cfg.PICAAD.PRIOR.SEED,
        )
    raise ValueError(f'Unknown PICAAD.PRIOR.TYPE: {prior_type}')


def apply_prior_to_model(cfg, model: PICAAD, te_weight_np, te_gate_np):
    model.set_te_prior(
        torch.from_numpy(te_weight_np),
        torch.from_numpy(te_gate_np),
        init_scale=cfg.PICAAD.PRIOR.INIT_SCALE,
    )


# ----------------------------------------------------------
# On-disk prior cache
# ----------------------------------------------------------
def _prior_cache_key(cfg, train_TN: np.ndarray, entity_name: str) -> str:
    """Deterministic cache key from prior hyperparameters + training data.

    Any change to hyperparameters, tau_max, or the training tensor content
    invalidates the cache. Excludes irrelevant params (BLEND, INIT_SCALE are
    applied at model-init time and don't affect the prior arrays).
    """
    h = hashlib.sha256()
    p = cfg.PICAAD.PRIOR
    parts = [
        cfg.DATA.NAME, entity_name, p.TYPE,
        f'tau{cfg.PICAAD.TAU_MAX}',
        f'self{p.SELF_MASS:g}', f'sd{p.SEED}',
    ]
    if p.TYPE == 'pcmci':
        parts += [p.PCMCI.CI_TEST, f'a{p.PCMCI.ALPHA:g}', f'sub{p.PCMCI.SUBSAMPLE}']
    else:
        parts += [f'b{p.TE.BINS}', f'nc{p.TE.NUM_CHUNKS}',
                  f'cl{p.TE.CHUNK_LEN}', f'th{p.TE.THRESHOLD:g}']
    h.update('|'.join(parts).encode('utf-8'))
    h.update(np.ascontiguousarray(train_TN).tobytes())
    digest = h.hexdigest()[:16]
    return f'{cfg.DATA.NAME}_{entity_name}_{p.TYPE}_{digest}'


def build_causal_prior_cached(cfg, train_TN: np.ndarray, entity_name: str):
    """Wraps build_causal_prior with an on-disk NPZ cache keyed by
    hyperparameters and the raw training tensor contents.
    """
    p = cfg.PICAAD.PRIOR
    if not p.CACHE_ENABLE:
        return build_causal_prior(cfg, train_TN)

    key = _prior_cache_key(cfg, train_TN, entity_name)
    cache_path = os.path.join(p.CACHE_DIR, f'{key}.npz')

    if not p.CACHE_REBUILD and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            te_weight_np = data['te_weight']
            te_gate_np = data['te_gate']
            print(f'  [prior] cache hit: {cache_path}', flush=True)
            return te_weight_np, te_gate_np
        except Exception as e:
            print(f'  [prior] cache read failed ({e}); rebuilding', flush=True)

    t0 = time.time()
    te_weight_np, te_gate_np = build_causal_prior(cfg, train_TN)
    dt = time.time() - t0

    os.makedirs(p.CACHE_DIR, exist_ok=True)
    try:
        np.savez(cache_path, te_weight=te_weight_np, te_gate=te_gate_np)
        print(f'  [prior] cached ({dt:.1f}s): {cache_path}', flush=True)
    except Exception as e:
        print(f'  [prior] cache write failed ({e}); continuing without cache', flush=True)

    return te_weight_np, te_gate_np
