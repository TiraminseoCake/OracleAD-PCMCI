"""PicaadTrainer: encapsulates the train loop, per-epoch eval, and reference
tensor (w_ref/cls_ref) maintenance for a single (entity, seed) run.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from datasets.loader import get_train_dataloader
from layers.ops import (
    make_self_causal_fallback_torch,
    normalize_causal_tensor_torch,
)
from model.intervention import permutation_alignment_and_epoch_cls
from model.losses import (
    causal_structure_loss,
    graph_stability_loss,
    invariance_loss_from_tensor,
    prediction_train_loss,
    reconstruction_train_loss,
)
from utils.evaluation import run_epoch_eval


# Group sub-weights (kept fixed as in the original picaad.py to preserve
# the achieved SWaT results; expose to cfg only if a future ablation needs it).
_W_RECON = 0.0    # reconstruction removed
_W_TE_GATE = 0.50
_W_GRAPH = 0.50
_W_LAGMONO = 0.50
_W_INV = 0.50


class PicaadTrainer:
    def __init__(self, cfg, model, entity, seed,
                 device, writer=None, ckpt_dir=None, csv_path=None):
        self.cfg = cfg
        self.model = model
        self.entity = entity          # datasets.build.EntityArrays
        self.seed = int(seed)
        self.device = device
        self.writer = writer
        self.writer_prefix = entity.name
        self.ckpt_dir = ckpt_dir
        self.csv_path = csv_path

        self.rng = np.random.default_rng(self.seed)

        self.model.to(device)
        self.model.reset_refs()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        self.train_loader = get_train_dataloader(cfg, entity.train_z)

    def train(self):
        cfg = self.cfg
        Tlag = self.model.tau_max
        N = self.model.N
        epochs = cfg.SOLVER.MAX_EPOCH

        for ep in range(1, epochs + 1):
            self.model.train()
            self.model._current_epoch = ep

            w_sum = torch.zeros(Tlag, N, N, device=self.device)
            w_cnt = 0

            cls_sum = torch.zeros(Tlag, N, N, device=self.device)
            cls_cnt = torch.zeros(Tlag, N, 1, device=self.device)

            stats = self._empty_stats()
            steps = 0
            last_use_cstruct = False
            last_use_graph_loss = False

            for X, env in self.train_loader:
                X = X.to(self.device)
                env = torch.as_tensor(env, device=self.device, dtype=torch.long)

                out = self.model(X)
                (recon, pred, C_all, pred_weights,
                 edge_value, edge_effect, edge_strength, gate, local_delta) = out

                xL = X[:, -1, :]
                xpast = X[:, :self.model.L - 1, :]

                loss_pred = prediction_train_loss(xL, pred, loss_type=cfg.PICAAD.TRAIN_LOSS_TYPE)
                loss_recon = reconstruction_train_loss(xpast, recon, loss_type=cfg.PICAAD.RECON_LOSS_TYPE)

                if pred_weights.dim() == 4:
                    w_epoch_mean = pred_weights.mean(dim=0).detach()
                else:
                    w_epoch_mean = pred_weights.detach()
                w_sum += w_epoch_mean
                w_cnt += 1

                use_cstruct_loss = (ep >= cfg.PICAAD.START_CLS_EPOCH) and self.model.has_cls_ref
                use_graph_loss = (ep >= cfg.PICAAD.START_WREF_EPOCH) and self.model.has_w_ref
                last_use_cstruct = use_cstruct_loss
                last_use_graph_loss = use_graph_loss

                loss_cstruct = (
                    causal_structure_loss(edge_strength, self.model.cls_ref)
                    if use_cstruct_loss
                    else torch.tensor(0.0, device=self.device)
                )
                loss_graph = (
                    graph_stability_loss(pred_weights, self.model.w_ref)
                    if use_graph_loss
                    else torch.tensor(0.0, device=self.device)
                )

                loss_gate = self.model.gate_sparsity()
                loss_lagmono = self.model.lag_monotonic_penalty()
                loss_te_w, loss_te_g = self.model.causal_prior_losses(pred_weights)

                base_abs_err_ref = (xL - pred).abs().detach()
                loss_perm, batch_cls_sum, batch_cls_cnt = permutation_alignment_and_epoch_cls(
                    self.model, X, xL, base_abs_err_ref, edge_strength, self.rng,
                    perm_pairs=cfg.PICAAD.INTERVENTION.PERM_PAIRS_PER_BATCH,
                    perm_mode=cfg.PICAAD.INTERVENTION.PERM_MODE,
                    fill_value=cfg.PICAAD.INTERVENTION.FILL_VALUE,
                )
                cls_sum += batch_cls_sum
                cls_cnt += batch_cls_cnt

                loss_inv = invariance_loss_from_tensor(edge_strength, env)

                group_task = loss_pred + _W_RECON * loss_recon
                group_causal = loss_te_w + _W_TE_GATE * loss_te_g
                if use_cstruct_loss:
                    group_causal = group_causal + loss_cstruct
                group_graphreg = loss_gate + _W_LAGMONO * loss_lagmono
                if use_graph_loss:
                    group_graphreg = group_graphreg + _W_GRAPH * loss_graph
                group_robust = loss_perm + _W_INV * loss_inv

                loss = (
                    cfg.PICAAD.LAM_TASK * group_task
                    + cfg.PICAAD.LAM_CAUSAL * group_causal
                    + cfg.PICAAD.LAM_GRAPHREG * group_graphreg
                    + cfg.PICAAD.LAM_ROBUST * group_robust
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.SOLVER.GRADIENT_CLIP and cfg.SOLVER.GRADIENT_CLIP > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRADIENT_CLIP)
                self.optimizer.step()

                self._accumulate_stats(stats, group_task, group_causal, group_graphreg,
                                        group_robust, loss, loss_pred, loss_recon,
                                        loss_te_w, loss_te_g, loss_cstruct, loss_graph,
                                        loss_gate, loss_lagmono, loss_perm, loss_inv)
                steps += 1

            self._update_epoch_refs(w_sum, w_cnt, cls_sum, cls_cnt)

            avg = {k: v / max(steps, 1) for k, v in stats.items()}
            self._print_epoch(ep, avg, last_use_cstruct, last_use_graph_loss)
            self._log_train_tb(ep, avg)

            self._maybe_run_epoch_eval(ep, epochs)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _empty_stats():
        return {
            "task": 0.0, "causal": 0.0, "graphreg": 0.0, "robust": 0.0, "total": 0.0,
            "pred": 0.0, "recon": 0.0, "tew": 0.0, "teg": 0.0,
            "cstruct": 0.0, "graph": 0.0, "gate": 0.0, "lagmono": 0.0,
            "perm": 0.0, "inv": 0.0,
        }

    @staticmethod
    def _accumulate_stats(stats, gt, gc, gg, gr, total, lp, lrc, ltew, lteg,
                          lc, lg, lgate, llm, lperm, linv):
        stats["task"]     += float(gt.detach().cpu())
        stats["causal"]   += float(gc.detach().cpu())
        stats["graphreg"] += float(gg.detach().cpu())
        stats["robust"]   += float(gr.detach().cpu())
        stats["total"]    += float(total.detach().cpu())
        stats["pred"]     += float(lp.detach().cpu())
        stats["recon"]    += float(lrc.detach().cpu())
        stats["tew"]      += float(ltew.detach().cpu())
        stats["teg"]      += float(lteg.detach().cpu())
        stats["cstruct"]  += float(lc.detach().cpu())
        stats["graph"]    += float(lg.detach().cpu())
        stats["gate"]     += float(lgate.detach().cpu())
        stats["lagmono"]  += float(llm.detach().cpu())
        stats["perm"]     += float(lperm.detach().cpu())
        stats["inv"]      += float(linv.detach().cpu())

    def _update_epoch_refs(self, w_sum, w_cnt, cls_sum, cls_cnt):
        cfg = self.cfg
        Tlag = self.model.tau_max
        N = self.model.N
        with torch.no_grad():
            epoch_w = w_sum / max(w_cnt, 1)
            if not self.model.has_w_ref:
                self.model.w_ref.copy_(epoch_w)
                self.model.has_w_ref = True
            elif cfg.PICAAD.WREF_EMA <= 0.0:
                self.model.w_ref.copy_(epoch_w)
            else:
                beta = float(cfg.PICAAD.WREF_EMA)
                self.model.w_ref.mul_(beta).add_(epoch_w * (1.0 - beta))

            if self.model.has_cls_ref:
                fallback_cls = self.model.cls_ref
            elif self.model.has_te_prior:
                fallback_cls = self.model.te_prior_weight
            else:
                fallback_cls = make_self_causal_fallback_torch(
                    Tlag, N, device=self.device, dtype=cls_sum.dtype
                )

            epoch_cls_raw = torch.where(
                cls_cnt > 0,
                cls_sum / cls_cnt.clamp_min(1.0),
                fallback_cls,
            )
            epoch_cls = normalize_causal_tensor_torch(epoch_cls_raw)

            if not self.model.has_cls_ref:
                self.model.cls_ref.copy_(epoch_cls)
                self.model.has_cls_ref = True
            elif cfg.PICAAD.CLS_EMA <= 0.0:
                self.model.cls_ref.copy_(epoch_cls)
            else:
                beta = float(cfg.PICAAD.CLS_EMA)
                self.model.cls_ref.mul_(beta).add_(epoch_cls * (1.0 - beta))
                self.model.cls_ref.copy_(normalize_causal_tensor_torch(self.model.cls_ref))

    def _print_epoch(self, ep, avg, last_use_cstruct, last_use_graph_loss):
        print(
            f"  [ep {ep:02d}] "
            f"task={avg['task']:.6f} causal={avg['causal']:.6f} "
            f"graphreg={avg['graphreg']:.6f} robust={avg['robust']:.6f} "
            f"total={avg['total']:.6f} | "
            f"pred={avg['pred']:.6f} recon={avg['recon']:.6f} "
            f"tew={avg['tew']:.6f} teg={avg['teg']:.6f} "
            f"cstruct={avg['cstruct']:.6f} graph={avg['graph']:.6f} "
            f"gate={avg['gate']:.6f} lagmono={avg['lagmono']:.6f} "
            f"perm={avg['perm']:.6f} inv={avg['inv']:.6f} "
            f"has_cls={self.model.has_cls_ref} use_cstruct={last_use_cstruct} "
            f"has_wref={self.model.has_w_ref} use_graph={last_use_graph_loss}",
            flush=True,
        )

    def _log_train_tb(self, ep, avg):
        if self.writer is None:
            return
        w = self.writer
        p = self.writer_prefix
        w.add_scalar(f"{p}/train/group_task",     avg["task"],     ep)
        w.add_scalar(f"{p}/train/group_causal",   avg["causal"],   ep)
        w.add_scalar(f"{p}/train/group_graphreg", avg["graphreg"], ep)
        w.add_scalar(f"{p}/train/group_robust",   avg["robust"],   ep)
        w.add_scalar(f"{p}/train/total_loss",     avg["total"],    ep)
        w.add_scalar(f"{p}/train/pred_loss",         avg["pred"],    ep)
        w.add_scalar(f"{p}/train/recon_loss",        avg["recon"],   ep)
        w.add_scalar(f"{p}/train/te_weight_loss",    avg["tew"],     ep)
        w.add_scalar(f"{p}/train/te_gate_loss",      avg["teg"],     ep)
        w.add_scalar(f"{p}/train/causal_struct_loss",avg["cstruct"], ep)
        w.add_scalar(f"{p}/train/graph_loss",        avg["graph"],   ep)
        w.add_scalar(f"{p}/train/gate_loss",         avg["gate"],    ep)
        w.add_scalar(f"{p}/train/lagmono_loss",      avg["lagmono"], ep)
        w.add_scalar(f"{p}/train/perm_loss",         avg["perm"],    ep)
        w.add_scalar(f"{p}/train/inv_loss",          avg["inv"],     ep)

        with torch.no_grad():
            gate_np = self.model.edge_gate().detach().cpu().numpy()
            pw_global = self.model.get_pred_weights().detach().cpu().numpy()
            w.add_scalar(f"{p}/train/gate_mean",           float(gate_np.mean()),  ep)
            w.add_scalar(f"{p}/train/gate_max",            float(gate_np.max()),   ep)
            w.add_scalar(f"{p}/train/pred_weight_mean",    float(pw_global.mean()), ep)
            w.add_scalar(f"{p}/train/pred_weight_max",     float(pw_global.max()),  ep)
            w.add_scalar(f"{p}/train/pred_weight_entropy",
                         float(self.model.pred_weight_entropy().detach().cpu()), ep)
            w.add_scalar(f"{p}/train/w_ref_mean",   float(self.model.w_ref.mean().detach().cpu()),   ep)
            w.add_scalar(f"{p}/train/cls_ref_mean", float(self.model.cls_ref.mean().detach().cpu()), ep)
            w.add_scalar(f"{p}/train/cls_ref_std",  float(self.model.cls_ref.std().detach().cpu()),  ep)

        if ep % 10 == 0:
            w.add_histogram(f"{p}/train/gate_hist",        gate_np,  ep)
            w.add_histogram(f"{p}/train/pred_weight_hist", pw_global, ep)
            w.add_histogram(f"{p}/train/w_ref_hist",   self.model.w_ref.detach().cpu().numpy(),   ep)
            w.add_histogram(f"{p}/train/cls_ref_hist", self.model.cls_ref.detach().cpu().numpy(), ep)

    def _maybe_run_epoch_eval(self, ep, epochs_total):
        every = self.cfg.TRAIN.EVAL_EVERY
        if every is None or every <= 0:
            return
        if (ep % every != 0) and (ep != epochs_total):
            return
        run_epoch_eval(
            self.model, self.entity.test_z, self.entity.y, self.device, self.cfg,
            ep=ep, seed=self.seed, name=self.entity.name,
            epochs_total=epochs_total,
            train_TN=self.entity.train_z,
            writer=self.writer, writer_prefix=self.writer_prefix,
            ckpt_dir=self.ckpt_dir, csv_path=self.csv_path,
            mu=self.entity.mu, sd=self.entity.sd,
        )
