"""Causal prior builders for PICAAD: TE (unconditional), CTE (conditional two-stage),
and PCMCI+ (from tigramite). All return (te_weight, te_gate) with shapes
[tau_max, N, N] and consistent normalization semantics.
"""
import numpy as np


# ============================================================
# TE causal prior (original, unconditional)
# ============================================================
def _fit_equal_width_bins(series_TN: np.ndarray, num_bins: int):
    T, N = series_TN.shape
    edges = []
    for i in range(N):
        col = series_TN[:, i].astype(np.float64)
        col = col[np.isfinite(col)]
        if col.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(col.min()), float(col.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
        edges.append(np.linspace(lo, hi, int(num_bins) + 1)[1:-1])
    return edges


def _digitize_with_edges(series_TN: np.ndarray, edges):
    T, N = series_TN.shape
    out = np.zeros((T, N), dtype=np.int64)
    for i, e in enumerate(edges):
        out[:, i] = np.digitize(series_TN[:, i], e, right=False)
    return out


def _transfer_entropy_discrete_1lag(x_disc: np.ndarray,
                                    y_disc: np.ndarray,
                                    tau: int,
                                    num_bins: int,
                                    eps: float = 1e-12) -> float:
    tau = int(tau)
    B = int(num_bins)
    T = len(x_disc)

    t0 = max(tau, 1)
    if T - t0 <= 1:
        return 0.0

    y_t = y_disc[t0:]
    y_prev = y_disc[t0 - 1:T - 1]
    x_prev = x_disc[t0 - tau:T - tau]

    M = int(y_t.shape[0])
    if M <= 1:
        return 0.0

    xyz_code = (y_t * B + y_prev) * B + x_prev
    yz_code = y_prev * B + x_prev
    yy_code = y_t * B + y_prev
    y_code = y_prev

    c_xyz = np.bincount(xyz_code, minlength=B * B * B).astype(np.float64)
    c_yz = np.bincount(yz_code, minlength=B * B).astype(np.float64)
    c_yy = np.bincount(yy_code, minlength=B * B).astype(np.float64)
    c_y = np.bincount(y_code, minlength=B).astype(np.float64)

    nz = np.flatnonzero(c_xyz > 0)
    if nz.size == 0:
        return 0.0

    yt = nz // (B * B)
    rem = nz % (B * B)
    yp = rem // B
    xp = rem % B

    num = c_xyz[nz] * c_y[yp]
    den = c_yz[yp * B + xp] * c_yy[yt * B + yp]

    te_nat = np.sum((c_xyz[nz] / float(M)) * np.log((num + eps) / (den + eps)))
    te_bits = te_nat / np.log(2.0)
    return float(max(te_bits, 0.0))


# ============================================================
# [FIX 3] Two-stage Conditional TE prior
#
# Problem with v1: correlation-based confounder selection confuses
# mediators (X->M->Y) with confounders (Z->X, Z->Y).
# Conditioning on a mediator blocks the real causal path.
#
# Solution: 2-stage approach
#   Stage 1: Compute unconditional TE for all pairs -> rough graph
#   Stage 2: For each (src, tgt), identify candidate confounders
#            as variables z with high TE *to both* src and tgt
#            (i.e. z is a common parent, not a descendant of src).
#            A mediator has high TE *from* src, so it's excluded.
#   Also: reduce bins for CTE (B_cond < B) to mitigate B^4 sparsity.
# ============================================================
def _conditional_te_discrete_1lag(x_disc: np.ndarray,
                                   y_disc: np.ndarray,
                                   z_disc: np.ndarray,
                                   tau: int,
                                   num_bins_xy: int,
                                   num_bins_z: int,
                                   eps: float = 1e-12) -> float:
    """
    CTE_{x->y|z}(tau).
    Uses separate bin counts for (x,y) vs z to reduce sparsity:
      x,y use num_bins_xy bins; z uses num_bins_z bins (typically smaller).
    Total bin combinations: Bxy^2 * Bz * Bxy = Bxy^3 * Bz (not Bxy^4).
    """
    tau = int(tau)
    Bxy = int(num_bins_xy)
    Bz = int(num_bins_z)
    T = len(x_disc)
    t0 = max(tau, 1)
    if T - t0 <= 1:
        return 0.0

    y_t    = y_disc[t0:]
    y_prev = y_disc[t0 - 1:T - 1]
    x_prev = x_disc[t0 - tau:T - tau]
    # re-bin z to fewer bins to reduce sparsity
    z_prev = np.clip(z_disc[t0 - 1:T - 1] * Bz // max(Bxy, 1), 0, Bz - 1).astype(np.int64)

    M = int(y_t.shape[0])
    if M <= 1:
        return 0.0

    # joint codes with mixed bin sizes
    cond_code    = y_prev * Bz + z_prev                                  # (y_{t-1}, z_{t-1})
    full_code    = ((y_t * Bxy + y_prev) * Bz + z_prev) * Bxy + x_prev  # (y_t, y_{t-1}, z_{t-1}, x_{t-tau})
    cond_x_code  = cond_code * Bxy + x_prev                             # (y_{t-1}, z_{t-1}, x_{t-tau})
    yt_cond_code = (y_t * Bxy + y_prev) * Bz + z_prev                   # (y_t, y_{t-1}, z_{t-1})

    S_full    = Bxy * Bxy * Bz * Bxy
    S_cond    = Bxy * Bz
    S_cond_x  = Bxy * Bz * Bxy
    S_yt_cond = Bxy * Bxy * Bz

    c_full    = np.bincount(full_code,    minlength=S_full).astype(np.float64)
    c_cond    = np.bincount(cond_code,    minlength=S_cond).astype(np.float64)
    c_cond_x  = np.bincount(cond_x_code,  minlength=S_cond_x).astype(np.float64)
    c_yt_cond = np.bincount(yt_cond_code, minlength=S_yt_cond).astype(np.float64)

    nz = np.flatnonzero(c_full > 0)
    if nz.size == 0:
        return 0.0

    # decode indices
    rem = nz.copy()
    yt_idx = rem // (Bxy * Bz * Bxy); rem = rem % (Bxy * Bz * Bxy)
    yp_idx = rem // (Bz * Bxy);       rem = rem % (Bz * Bxy)
    zp_idx = rem // Bxy
    xp_idx = rem % Bxy

    cond_idx    = yp_idx * Bz + zp_idx
    cond_x_idx  = cond_idx * Bxy + xp_idx
    yt_cond_idx = (yt_idx * Bxy + yp_idx) * Bz + zp_idx

    num = c_full[nz] * c_cond[cond_idx]
    den = c_cond_x[cond_x_idx] * c_yt_cond[yt_cond_idx]

    cte_nat = np.sum((c_full[nz] / float(M)) * np.log((num + eps) / (den + eps)))
    cte_bits = cte_nat / np.log(2.0)
    return float(max(cte_bits, 0.0))


def _find_confounders_from_rough_graph(rough_te: np.ndarray, src: int, tgt: int,
                                        confounder_thresh: float = 0.1) -> list:
    """
    Identify candidate confounders for (src->tgt) using a rough TE graph.

    A confounder z has:  z->src (high TE) AND z->tgt (high TE)
    A mediator m has:    src->m (high TE) AND m->tgt (high TE)

    We select z where TE(z->src) and TE(z->tgt) are both above threshold,
    but TE(src->z) is NOT high (excludes mediators/descendants).

    rough_te: [N, N] where rough_te[i, j] = max over tau of TE(i->j)
    Returns list of z indices (can be empty).
    """
    N = rough_te.shape[0]
    if N < 3:
        return []

    candidates = []
    for z in range(N):
        if z == src or z == tgt:
            continue
        # z -> src and z -> tgt must be strong (common parent pattern)
        te_z_to_src = rough_te[z, src]
        te_z_to_tgt = rough_te[z, tgt]
        # src -> z must be weak (exclude mediators/descendants)
        te_src_to_z = rough_te[src, z]

        if (te_z_to_src > confounder_thresh and
            te_z_to_tgt > confounder_thresh and
            te_src_to_z < te_z_to_src * 0.5):  # z causes src, not the reverse
            score = te_z_to_src + te_z_to_tgt  # confounding strength
            candidates.append((z, score))

    # return top-1 confounder (strongest common parent)
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [candidates[0][0]]


def build_cte_causal_prior(train_TN: np.ndarray,
                           tau_max: int,
                           num_bins: int = 8,
                           num_chunks: int = 32,
                           chunk_len: int = 256,
                           threshold: float = 0.0,
                           self_mass: float = 0.25,
                           seed: int = 0):
    """
    [FIX 3] Two-stage Conditional TE prior.

    Stage 1: Compute unconditional TE to get rough directed graph.
    Stage 2: For each (src, tgt) pair, use the rough graph to identify
             true confounders (common parents) vs mediators, then
             compute CTE conditioning only on confounders.

    Key improvements over v1:
    - Confounder selection uses causal structure, not correlation
    - Reduced z-bins (num_bins//2) to mitigate B^4 sparsity
    - Falls back to unconditional TE when no confounder is found

    Returns:
        te_weight: [tau_max, N, N]
        te_gate:   [tau_max, N, N]
    """
    train_TN = np.asarray(train_TN, dtype=np.float32)
    T, N = train_TN.shape
    tau_max = int(tau_max)
    num_bins_z = max(num_bins // 2, 2)  # fewer bins for conditioning var

    if T < tau_max + 3:
        raise ValueError(f"train length too short for tau_max={tau_max}: T={T}")

    edges = _fit_equal_width_bins(train_TN, num_bins=num_bins)
    disc_all = _digitize_with_edges(train_TN, edges)

    min_len = max(tau_max + 3, 16)
    if T <= max(int(chunk_len), min_len):
        starts = [0]
        actual_len = T
    else:
        rng = np.random.default_rng(seed)
        actual_len = max(min(int(chunk_len), T), min_len)
        max_start = T - actual_len
        starts = rng.integers(0, max_start + 1, size=max(int(num_chunks), 1)).tolist()

    # ---- Stage 1: unconditional TE -> rough graph ----
    te_stage1 = np.zeros((tau_max, N, N), dtype=np.float64)
    used = 0
    for s in starts:
        seg = disc_all[s:s + actual_len]
        if seg.shape[0] < min_len:
            continue
        used += 1
        for tau in range(1, tau_max + 1):
            for src in range(N):
                x = seg[:, src]
                for tgt in range(N):
                    if src == tgt:
                        continue
                    y = seg[:, tgt]
                    te_stage1[tau - 1, src, tgt] += _transfer_entropy_discrete_1lag(
                        x, y, tau=tau, num_bins=num_bins
                    )
    if used == 0:
        raise RuntimeError("No valid chunks for CTE stage-1.")
    te_stage1 /= float(used)

    # rough graph: max TE over lags for each (src, tgt)
    rough_te_max = te_stage1.max(axis=0)  # [N, N]
    # adaptive threshold: median of nonzero values
    nz_vals = rough_te_max[rough_te_max > 0]
    conf_thresh = float(np.median(nz_vals)) if nz_vals.size > 0 else 0.1

    print(f"  [CTE] stage-1 done: rough graph density="
          f"{(rough_te_max > conf_thresh).sum()}/{N*N}, "
          f"confounder_thresh={conf_thresh:.4f}", flush=True)

    # ---- Stage 2: CTE conditioning on identified confounders ----
    # pre-compute confounder map
    confounder_map = {}
    for src in range(N):
        for tgt in range(N):
            if src == tgt:
                continue
            confounder_map[(src, tgt)] = _find_confounders_from_rough_graph(
                rough_te_max, src, tgt, confounder_thresh=conf_thresh
            )

    n_conditioned = sum(1 for v in confounder_map.values() if len(v) > 0)
    print(f"  [CTE] confounder map: {n_conditioned}/{len(confounder_map)} pairs have confounders",
          flush=True)

    te_acc = np.zeros((tau_max, N, N), dtype=np.float64)
    used2 = 0
    for s in starts:
        seg = disc_all[s:s + actual_len]
        if seg.shape[0] < min_len:
            continue
        used2 += 1
        for tau in range(1, tau_max + 1):
            for src in range(N):
                x = seg[:, src]
                for tgt in range(N):
                    if src == tgt:
                        continue
                    y = seg[:, tgt]
                    confounders = confounder_map[(src, tgt)]

                    if confounders:
                        z_idx = confounders[0]
                        z = seg[:, z_idx]
                        val = _conditional_te_discrete_1lag(
                            x, y, z, tau=tau,
                            num_bins_xy=num_bins,
                            num_bins_z=num_bins_z,
                        )
                    else:
                        # no confounder identified -> use unconditional TE
                        val = _transfer_entropy_discrete_1lag(
                            x, y, tau=tau, num_bins=num_bins
                        )
                    te_acc[tau - 1, src, tgt] += val

    if used2 == 0:
        raise RuntimeError("No valid chunks for CTE stage-2.")

    te_raw = te_acc / float(used2)
    te_raw[te_raw < float(threshold)] = 0.0

    diag = np.arange(N)
    te_raw[0, diag, diag] = np.maximum(te_raw[0, diag, diag], float(self_mass))

    flat = te_raw.reshape(tau_max * N, N)
    colsum = flat.sum(axis=0, keepdims=True)

    fallback = np.zeros_like(flat)
    fallback[diag, diag] = 1.0

    flat = np.where(colsum > 1e-12, flat / np.clip(colsum, 1e-12, None), fallback)
    te_weight = flat.reshape(tau_max, N, N).astype(np.float32)

    nz = te_raw[te_raw > 0]
    scale = float(np.quantile(nz, 0.75)) if nz.size > 0 else 1.0
    te_gate = np.clip(te_raw / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)
    te_gate[0, diag, diag] = 1.0

    return te_weight, te_gate


def build_te_causal_prior(train_TN: np.ndarray,
                          tau_max: int,
                          num_bins: int = 8,
                          num_chunks: int = 32,
                          chunk_len: int = 256,
                          threshold: float = 0.0,
                          self_mass: float = 0.25,
                          seed: int = 0):
    train_TN = np.asarray(train_TN, dtype=np.float32)
    T, N = train_TN.shape
    tau_max = int(tau_max)

    if T < tau_max + 3:
        raise ValueError(f"train length too short for tau_max={tau_max}: T={T}")

    edges = _fit_equal_width_bins(train_TN, num_bins=num_bins)
    disc_all = _digitize_with_edges(train_TN, edges)

    te_acc = np.zeros((tau_max, N, N), dtype=np.float64)

    min_len = max(tau_max + 3, 16)
    if T <= max(int(chunk_len), min_len):
        starts = [0]
        actual_len = T
    else:
        rng = np.random.default_rng(seed)
        actual_len = max(min(int(chunk_len), T), min_len)
        max_start = T - actual_len
        starts = rng.integers(0, max_start + 1, size=max(int(num_chunks), 1)).tolist()

    used = 0
    for s in starts:
        seg = disc_all[s:s + actual_len]
        if seg.shape[0] < min_len:
            continue
        used += 1
        for tau in range(1, tau_max + 1):
            for src in range(N):
                x = seg[:, src]
                for tgt in range(N):
                    if src == tgt:
                        continue
                    y = seg[:, tgt]
                    te_acc[tau - 1, src, tgt] += _transfer_entropy_discrete_1lag(
                        x, y, tau=tau, num_bins=num_bins
                    )

    if used == 0:
        raise RuntimeError("No valid chunks were available for TE prior estimation.")

    te_raw = te_acc / float(used)
    te_raw[te_raw < float(threshold)] = 0.0

    diag = np.arange(N)
    te_raw[0, diag, diag] = np.maximum(te_raw[0, diag, diag], float(self_mass))

    flat = te_raw.reshape(tau_max * N, N)
    colsum = flat.sum(axis=0, keepdims=True)
    fallback = np.zeros_like(flat)
    fallback[diag, diag] = 1.0
    flat = np.where(colsum > 1e-12, flat / np.clip(colsum, 1e-12, None), fallback)
    te_weight = flat.reshape(tau_max, N, N).astype(np.float32)

    nz = te_raw[te_raw > 0]
    scale = float(np.quantile(nz, 0.75)) if nz.size > 0 else 1.0
    te_gate = np.clip(te_raw / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)
    te_gate[0, diag, diag] = 1.0

    return te_weight, te_gate


# ============================================================
# PCMCI+ causal prior
# ============================================================
def build_pcmci_causal_prior(train_TN: np.ndarray,
                              tau_max: int,
                              ci_test: str = "ParCorr",
                              pc_alpha: float = 0.05,
                              subsample: int = 10000,
                              self_mass: float = 0.25,
                              seed: int = 0):
    """
    Use PCMCI+ from tigramite for causal discovery.
    Produces te_weight and te_gate with same interface as build_te_causal_prior.

    PCMCI+ advantages over TE:
      - Systematic conditional independence testing
      - Handles contemporaneous + lagged effects
      - Removes confounders by conditioning on full parent set

    Args:
        train_TN: [T, N] training data
        tau_max: maximum lag
        ci_test: "ParCorr" (linear, fast) or "CMIknn" (nonlinear, slow)
        pc_alpha: significance level for edge pruning
        subsample: max samples to use (for speed)
        self_mass: minimum self-loop weight
        seed: random seed for subsampling
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    if ci_test == "ParCorr":
        from tigramite.independence_tests.parcorr import ParCorr
        cond_ind_test = ParCorr(significance='analytic')
    elif ci_test == "CMIknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        cond_ind_test = CMIknn()
    else:
        raise ValueError(f"Unknown ci_test: {ci_test}")

    train_TN = np.asarray(train_TN, dtype=np.float64)
    T, N = train_TN.shape

    # Subsample for speed (PCMCI+ on T=500K is too slow)
    if T > subsample:
        rng = np.random.default_rng(seed)
        # Take contiguous block to preserve temporal structure
        start = rng.integers(0, T - subsample)
        train_sub = train_TN[start:start + subsample]
        print(f"  [PCMCI+] Subsampled T={T} -> {subsample} (start={start})", flush=True)
    else:
        train_sub = train_TN

    var_names = [f"v{i}" for i in range(N)]
    dataframe = pp.DataFrame(train_sub, var_names=var_names)

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    print(f"  [PCMCI+] Running PCMCI+ (N={N}, T={len(train_sub)}, tau_max={tau_max}, "
          f"ci_test={ci_test}, alpha={pc_alpha}) ...", flush=True)

    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)

    # results['val_matrix']: [N, N, tau_max+1] - test statistic values
    # results['p_matrix']:   [N, N, tau_max+1] - p-values
    # results['graph']:      [N, N, tau_max+1] - edge types ('-->', '<--', 'o-o', '', etc.)

    val_matrix = results['val_matrix']   # [N, N, tau_max+1]
    p_matrix = results['p_matrix']       # [N, N, tau_max+1]
    graph = results['graph']             # [N, N, tau_max+1] string array

    # Convert to te_weight [tau_max, N, N] and te_gate [tau_max, N, N]
    # Use lagged effects: tau=1..tau_max (index 1..tau_max in results)
    te_raw = np.zeros((tau_max, N, N), dtype=np.float64)
    te_gate_raw = np.zeros((tau_max, N, N), dtype=np.float32)

    for tau_idx in range(1, tau_max + 1):
        for i in range(N):
            for j in range(N):
                edge_type = graph[i, j, tau_idx]
                p_val = p_matrix[i, j, tau_idx]
                val = abs(val_matrix[i, j, tau_idx])

                # '-->' means i(t-tau) -> j(t) is a directed causal link
                if edge_type == '-->':
                    te_raw[tau_idx - 1, i, j] = val
                    te_gate_raw[tau_idx - 1, i, j] = 1.0
                # 'o-o' or 'x-x' means ambiguous - include with lower confidence
                elif edge_type in ('o-o', 'x-x'):
                    te_raw[tau_idx - 1, i, j] = val * 0.5
                    te_gate_raw[tau_idx - 1, i, j] = 0.5

    # Also handle contemporaneous effects (tau=0) -> put in tau=1 slot
    for i in range(N):
        for j in range(N):
            edge_type = graph[i, j, 0]
            val = abs(val_matrix[i, j, 0])
            if edge_type == '-->':
                # contemporaneous i -> j, add to tau=1
                te_raw[0, i, j] = max(te_raw[0, i, j], val)
                te_gate_raw[0, i, j] = max(te_gate_raw[0, i, j], 1.0)

    # Self-loop
    diag = np.arange(N)
    te_raw[0, diag, diag] = np.maximum(te_raw[0, diag, diag], float(self_mass))

    # Normalize te_weight over (tau, source) for each target
    flat = te_raw.reshape(tau_max * N, N)
    colsum = flat.sum(axis=0, keepdims=True)
    fallback = np.zeros_like(flat)
    fallback[diag, diag] = 1.0
    flat = np.where(colsum > 1e-12, flat / np.clip(colsum, 1e-12, None), fallback)
    te_weight = flat.reshape(tau_max, N, N).astype(np.float32)

    # Gate: already binary/soft from PCMCI+ results
    te_gate_raw[0, diag, diag] = 1.0
    te_gate = te_gate_raw

    n_directed = (te_gate == 1.0).sum()
    n_ambiguous = ((te_gate > 0) & (te_gate < 1.0)).sum()
    n_total = tau_max * N * N
    print(f"  [PCMCI+] Done. Directed edges: {n_directed}/{n_total}, "
          f"Ambiguous: {n_ambiguous}, Empty: {n_total - n_directed - n_ambiguous}", flush=True)

    return te_weight, te_gate
