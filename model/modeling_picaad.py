"""PICAAD (Causal Prior-Guided Anomaly Detection) model.

Per-Variable LSTM encoder + Temporal AttnPool -> MHSA (N variable nodes,
per lag) -> Edge-wise Prediction -> Scoring. PCMCI+/TE causal prior
initializes edge gate, pred_logits, and a learnable causal attention mask.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.encoder import PerVarEncoder
from layers.ops import logit_from_prob, normalize_causal_tensor_torch


class PICAAD(nn.Module):
    def __init__(self, N: int, L: int, tau_max: int, d: int, heads: int,
                 enc_layers: int, dec_layers: int, dropout: float,
                 mhsa_residual: bool = False,
                 lag_fusion: str = "mean",
                 lag_win: int = 5,
                 pred_temp: float = 1.0,
                 self_loop_bias: float = 1.0,
                 lag_source_topk: int = 0,
                 dynamic_graph: bool = True,
                 graph_hidden: int = 16,
                 gate_init: float = 0.15,
                 te_prior_blend: float = 0.35,
                 # [FIX 4] learnable causal attention mask
                 causal_attn_mask_scale: float = 0.5,
                 causal_mask_warmup_epochs: int = 5):
        super().__init__()
        self.N = N
        self.L = L
        self.tau_max = tau_max
        self.d = d
        self.lag_fusion = lag_fusion
        self.lag_win = int(lag_win)
        self.pred_temp = float(pred_temp)
        self.self_loop_bias = float(self_loop_bias)
        self.lag_source_topk = int(lag_source_topk)
        self.dynamic_graph = bool(dynamic_graph)
        self.graph_hidden = int(graph_hidden)
        self.te_prior_blend = float(te_prior_blend)
        self.causal_attn_mask_scale = float(causal_attn_mask_scale)
        self.causal_mask_warmup_epochs = int(causal_mask_warmup_epochs)
        self._current_epoch = 0  # set by training loop

        if tau_max >= L:
            raise ValueError(f"tau_max={tau_max} must be < L={L}")
        if self.lag_win <= 0:
            raise ValueError(f"lag_win must be >= 1, got {self.lag_win}")

        self.encoders = nn.ModuleList([PerVarEncoder(d, enc_layers, dropout) for _ in range(N)])
        self.mhsa = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        # Reconstruction removed - ablation showed no contribution to AD performance
        self.mhsa_residual = mhsa_residual

        self.pred_logits = nn.Parameter(torch.zeros(tau_max, N, N))
        with torch.no_grad():
            diag_idx = torch.arange(N)
            self.pred_logits[:, diag_idx, diag_idx] += self.self_loop_bias

        self.edge_log_alpha = nn.Parameter(torch.full((tau_max, N, N), logit_from_prob(gate_init)))

        if self.dynamic_graph:
            self.dynamic_q = nn.Linear(d, self.graph_hidden, bias=False)
            self.dynamic_k = nn.Linear(d, self.graph_hidden, bias=False)
            self.dynamic_scale = 1.0 / math.sqrt(max(self.graph_hidden, 1))

        self.edge_value_head = nn.Linear(d, N, bias=True)
        self.pred_bias = nn.Parameter(torch.zeros(N))

        # persistent references (not saved in state_dict flags - see register_buffer for tensors)
        self.register_buffer("cls_ref", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("w_ref", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("_has_cls_ref", torch.zeros(1, dtype=torch.bool), persistent=True)
        self.register_buffer("_has_w_ref", torch.zeros(1, dtype=torch.bool), persistent=True)

        self.register_buffer("te_prior_weight", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("te_prior_gate", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("_has_te_prior", torch.zeros(1, dtype=torch.bool), persistent=True)

        # [FIX 4] Learnable causal attention mask logits: [tau_max, N, N]
        # Initialized to zeros (uniform attention). set_te_prior() initializes
        # from TE gate. Then the model learns to refine the mask during training.
        self.causal_mask_logits = nn.Parameter(torch.zeros(tau_max, N, N))

    # Properties so callers can use model.has_cls_ref as before
    @property
    def has_cls_ref(self):
        return bool(self._has_cls_ref.item())

    @has_cls_ref.setter
    def has_cls_ref(self, v):
        self._has_cls_ref.fill_(int(bool(v)))

    @property
    def has_w_ref(self):
        return bool(self._has_w_ref.item())

    @has_w_ref.setter
    def has_w_ref(self, v):
        self._has_w_ref.fill_(int(bool(v)))

    @property
    def has_te_prior(self):
        return bool(self._has_te_prior.item())

    @has_te_prior.setter
    def has_te_prior(self, v):
        self._has_te_prior.fill_(int(bool(v)))

    def reset_refs(self):
        self.w_ref.zero_()
        self.has_w_ref = False

        if self.has_te_prior:
            self.cls_ref.copy_(self.te_prior_weight)
            self.has_cls_ref = True
        else:
            self.cls_ref.zero_()
            self.has_cls_ref = False

    def edge_gate(self):
        return torch.sigmoid(self.edge_log_alpha)

    def gate_sparsity(self):
        return self.edge_gate().mean()

    def lag_monotonic_penalty(self):
        if self.tau_max <= 1:
            return torch.tensor(0.0, device=self.edge_log_alpha.device)
        gate = self.edge_gate()
        return F.relu(gate[1:] - gate[:-1]).mean()

    @torch.no_grad()
    def set_te_prior(self, te_weight: torch.Tensor, te_gate: torch.Tensor = None, init_scale: float = 0.25):
        te_weight = te_weight.to(device=self.pred_logits.device, dtype=self.pred_logits.dtype).clamp_min(0.0)

        if te_gate is None:
            te_gate = (te_weight > 0).to(dtype=self.pred_logits.dtype)
        else:
            te_gate = te_gate.to(device=self.pred_logits.device, dtype=self.pred_logits.dtype).clamp(0.0, 1.0)

        te_weight = normalize_causal_tensor_torch(te_weight)
        diag = torch.arange(self.N, device=te_weight.device)
        te_gate[0, diag, diag] = 1.0

        self.te_prior_weight.copy_(te_weight)
        self.te_prior_gate.copy_(te_gate)
        self.has_te_prior = True

        self.cls_ref.copy_(te_weight)
        self.has_cls_ref = True

        if init_scale > 0.0:
            prior_score = torch.log(te_weight.clamp_min(1e-8))
            g = te_gate.clamp(1e-4, 1.0 - 1e-4)
            prior_alpha = torch.log(g / (1.0 - g))
            self.pred_logits.add_(float(init_scale) * prior_score)
            self.edge_log_alpha.add_(0.5 * float(init_scale) * prior_alpha)

        # [FIX 4] Initialize learnable causal mask from TE gate logits
        g = te_gate.clamp(1e-4, 1.0 - 1e-4)
        self.causal_mask_logits.data.copy_(torch.log(g / (1.0 - g)) * 0.5)

    def _effective_gate(self):
        gate = self.edge_gate()
        if not self.has_te_prior:
            return gate
        return gate * (0.05 + 0.95 * self.te_prior_gate)

    def _compute_dynamic_delta(self, C_all):
        if not self.dynamic_graph:
            return None
        q = self.dynamic_q(C_all)
        k = self.dynamic_k(C_all)
        delta = torch.einsum("btsh,btih->btsi", q, k) * self.dynamic_scale
        return delta

    def _normalize_weight_tensor(self, score, gate):
        tau_max, N, _ = self.pred_logits.shape
        temp = max(self.pred_temp, 1e-6)

        if score.dim() == 3:
            flat_s = (score / temp).reshape(tau_max * N, N)
            flat_g = gate.reshape(tau_max * N, N)
            flat_s = flat_s - flat_s.max(dim=0, keepdim=True).values
            unnorm = torch.exp(flat_s) * flat_g
            if self.lag_source_topk > 0:
                k = min(self.lag_source_topk, tau_max * N)
                _, idx = torch.topk(unnorm, k=k, dim=0)
                mask = torch.zeros_like(unnorm)
                mask.scatter_(0, idx, 1.0)
                unnorm = unnorm * mask
            weights = unnorm / (unnorm.sum(dim=0, keepdim=True) + 1e-12)
            return weights.view(tau_max, N, N)

        if score.dim() == 4:
            B = score.shape[0]
            flat_s = (score / temp).reshape(B, tau_max * N, N)
            flat_g = gate.reshape(1, tau_max * N, N)
            flat_s = flat_s - flat_s.max(dim=1, keepdim=True).values
            unnorm = torch.exp(flat_s) * flat_g
            if self.lag_source_topk > 0:
                k = min(self.lag_source_topk, tau_max * N)
                _, idx = torch.topk(unnorm, k=k, dim=1)
                mask = torch.zeros_like(unnorm)
                mask.scatter_(1, idx, 1.0)
                unnorm = unnorm * mask
            weights = unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-12)
            return weights.view(B, tau_max, N, N)

        raise ValueError(f"score dim must be 3 or 4, got {score.dim()}")

    def get_pred_weights(self, local_delta=None):
        gate = self._effective_gate()

        if self.has_te_prior and self.te_prior_blend > 0.0:
            prior_bias = self.te_prior_blend * torch.log(self.te_prior_weight.clamp_min(1e-8))
        else:
            prior_bias = 0.0

        if local_delta is None:
            score = self.pred_logits + prior_bias
        else:
            if torch.is_tensor(prior_bias):
                score = self.pred_logits.unsqueeze(0) + prior_bias.unsqueeze(0) + local_delta
            else:
                score = self.pred_logits.unsqueeze(0) + local_delta
        return self._normalize_weight_tensor(score, gate)

    def pred_weight_entropy(self, weights=None):
        if weights is None:
            weights = self.get_pred_weights()
        tau_max, N, _ = self.pred_logits.shape
        denom = max(math.log(max(tau_max * N, 2)), 1e-6)
        if weights.dim() == 3:
            flat = weights.reshape(tau_max * N, N)
            return -(flat * torch.log(flat + 1e-12)).sum(dim=0).mean() / denom
        if weights.dim() == 4:
            flat = weights.reshape(weights.shape[0], tau_max * N, N)
            return -(flat * torch.log(flat + 1e-12)).sum(dim=1).mean() / denom
        raise ValueError(f"weights dim must be 3 or 4, got {weights.dim()}")

    # --------------------------------------------------------
    # [FIX 1] JSD (Jensen-Shannon Divergence)
    # Unlike symmetric KL which is unbounded, JSD is a true metric
    # (bounded in [0, log2]), well-behaved when p and q have
    # different supports. JSD = 0.5*KL(p||m) + 0.5*KL(q||m), m=(p+q)/2
    # --------------------------------------------------------
    def causal_prior_losses(self, pred_weights=None):
        zero = torch.tensor(0.0, device=self.pred_logits.device)
        if not self.has_te_prior:
            return zero, zero

        if pred_weights is None:
            pred_weights = self.get_pred_weights()

        if pred_weights.dim() == 4:
            w_mean = pred_weights.mean(dim=0)
        else:
            w_mean = pred_weights

        p = w_mean.reshape(self.tau_max * self.N, self.N).clamp_min(1e-8)
        q = self.te_prior_weight.reshape(self.tau_max * self.N, self.N).clamp_min(1e-8)

        # JSD: bounded, symmetric, true metric
        m = 0.5 * (p + q)
        loss_te_w = 0.5 * (F.kl_div(m.log(), p, reduction="batchmean")
                         + F.kl_div(m.log(), q, reduction="batchmean"))

        loss_te_g = F.binary_cross_entropy(
            self.edge_gate().clamp(1e-4, 1.0 - 1e-4),
            self.te_prior_gate.clamp(1e-4, 1.0 - 1e-4),
        )
        return loss_te_w, loss_te_g

    # --------------------------------------------------------
    # [FIX 4] Learnable causal attention mask
    #
    # v1 problems fixed:
    #   1. Frozen prior -> now learnable (nn.Parameter, initialized from TE gate)
    #   2. Scale too aggressive -> warmup ramp (0->scale over warmup_epochs)
    #   3. Bad prior contaminates everything -> learnable logits can diverge
    #      from prior as model learns better structure
    #
    # mask[tgt, src] = warmup_scale * sigmoid(causal_mask_logits[tau, src, tgt])
    # sigmoid output in (0,1) -> converted to log-domain additive bias.
    # --------------------------------------------------------
    def _causal_attn_mask(self, tau: int):
        if self.causal_attn_mask_scale <= 0.0:
            return None

        # warmup: ramp from 0 to full scale over warmup epochs
        if self.causal_mask_warmup_epochs > 0 and self._current_epoch > 0:
            ramp = min(float(self._current_epoch) / float(self.causal_mask_warmup_epochs), 1.0)
        else:
            ramp = 0.0  # epoch 0 = no mask yet (let model learn basic representations first)

        if ramp <= 0.0:
            return None

        # learnable soft gate for this lag
        soft_gate = torch.sigmoid(self.causal_mask_logits[tau - 1])  # [src, tgt] in (0,1)
        # log-domain bias: mask[tgt, src] for PyTorch MHA convention
        log_bias = torch.log(soft_gate.clamp_min(1e-6)).T  # [tgt, src]
        return (self.causal_attn_mask_scale * ramp) * log_bias

    def forward(self, X, mask_tau=None, mask_var=None, mask_fill_value=0.0):
        B, L, N = X.shape
        lag_embeds = []

        for tau in range(1, self.tau_max + 1):
            end = L - tau
            start = max(0, end - self.lag_win)

            c_list = []
            for i in range(N):
                x_i = X[:, start:end, i].unsqueeze(-1)
                if (mask_tau is not None) and (mask_var is not None):
                    if (tau == int(mask_tau)) and (i == int(mask_var)):
                        x_i = torch.full_like(x_i, float(mask_fill_value))
                ci = self.encoders[i](x_i)
                c_list.append(ci)

            C_tau = torch.stack(c_list, dim=1)  # [B, src, d]

            # [FIX 4] apply causal mask to MHSA
            causal_mask = self._causal_attn_mask(tau)
            A_tau, _ = self.mhsa(C_tau, C_tau, C_tau,
                                  attn_mask=causal_mask,
                                  need_weights=False)
            C_star_tau = (C_tau + A_tau) if self.mhsa_residual else A_tau
            lag_embeds.append(C_star_tau)

        C_all = torch.stack(lag_embeds, dim=1)  # [B, tau, src, d]

        # Reconstruction removed - ablation showed no contribution
        recon = torch.zeros(B, self.L - 1, N, device=X.device)

        local_delta = self._compute_dynamic_delta(C_all)
        pred_weights = self.get_pred_weights(local_delta=local_delta)

        edge_value = self.edge_value_head(C_all)

        edge_effect = edge_value * pred_weights
        edge_strength = pred_weights * edge_value.abs()
        pred = edge_effect.sum(dim=(1, 2)) + self.pred_bias

        return recon, pred, C_all, pred_weights, edge_value, edge_effect, edge_strength, self.edge_gate(), local_delta
