"""Evaluation helpers: paper-style metric wrapper and per-epoch evaluation
that logs metrics to CSV/TB and saves a checkpoint (.pt).
"""
import os

import numpy as np
import pandas as pd
import torch

from datasets.util import get_median_anomaly_length
from metrics.paper_eval.metrics_api import get_metrics as paper_get_metrics
from model.scoring import (
    fit_score_calibrator,
    score_components_to_timeline,
    score_windows,
)
from utils.misc import pct


def paper_eval_one(score_series_1d, y01, start_idx, eval_cfg):
    """eval_cfg: cfg.EVAL subtree with attributes
        USE_MEDIAN_VUS_WINDOW, SLIDING_WINDOW, VUS_VERSION, VUS_THRE.
    """
    score = score_series_1d[start_idx:].astype(np.float64)
    labels = y01[start_idx:].astype(np.int32)

    m = (~np.isnan(score)) & np.isfinite(score)
    score = score[m]
    labels = labels[m]

    if score.size == 0:
        return {k: float("nan") for k in [
            "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
            "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
        ]}

    sliding_window = (
        get_median_anomaly_length(labels)
        if eval_cfg.USE_MEDIAN_VUS_WINDOW
        else eval_cfg.SLIDING_WINDOW
    )

    return paper_get_metrics(
        score=score, labels=labels, slidingWindow=sliding_window,
        pred=None, version=eval_cfg.VUS_VERSION, thre=eval_cfg.VUS_THRE,
    )


def run_epoch_eval(model, test_TN, y, device, cfg,
                   ep, seed, name, epochs_total,
                   train_TN=None,
                   writer=None, writer_prefix="",
                   ckpt_dir=None, csv_path=None,
                   mu=None, sd=None):
    """Run test-set evaluation at a given epoch. Saves .pt checkpoint,
    appends metrics row to csv_path, logs to TensorBoard.

    mu, sd: entity standardization stats to persist in the checkpoint so
    inference on unseen data can be done from the .pt alone."""
    was_training = model.training
    model.eval()

    calibrator = None
    if cfg.PICAAD.SCORING.CALIBRATE and train_TN is not None:
        train_scores = score_windows(model, train_TN, device,
                                     batch=cfg.TRAIN.BATCH_SIZE,
                                     scoring_cfg=cfg.PICAAD.SCORING,
                                     calibrator=None)
        calibrator = fit_score_calibrator(train_scores)

    test_scores = score_windows(model, test_TN, device,
                                batch=cfg.TEST.BATCH_SIZE,
                                scoring_cfg=cfg.PICAAD.SCORING,
                                calibrator=calibrator)

    Tt = test_TN.shape[0]
    start = cfg.PICAAD.L - 1
    score_t_dict = score_components_to_timeline(
        {k: test_scores[k] for k in ["P", "C", "G", "S", "A", "P_raw", "C_raw", "G_raw"]},
        Tt=Tt, start=start,
    )
    A_t = score_t_dict["A_t"]

    mtr_A = paper_eval_one(A_t, y, start, cfg.EVAL)

    A_PR   = float(mtr_A["AUC-PR"])
    A_ROC  = float(mtr_A["AUC-ROC"])
    VUS_PR = float(mtr_A["VUS-PR"])
    VUS_ROC= float(mtr_A["VUS-ROC"])
    F1     = float(mtr_A["Standard-F1"])
    PA_F1  = float(mtr_A["PA-F1"])
    EV_F1  = float(mtr_A["Event-based-F1"])
    R_F1   = float(mtr_A["R-based-F1"])
    Aff_F1 = float(mtr_A["Affiliation-F"])

    print(
        f"  [ep {ep:02d} eval seed {seed}] "
        f"A-PR={pct(A_PR):.2f}  A-ROC={pct(A_ROC):.2f}  "
        f"F1={pct(F1):.2f}  PA-F1={pct(PA_F1):.2f}  EV-F1={pct(EV_F1):.2f}  "
        f"R-F1={pct(R_F1):.2f}  Aff-F={pct(Aff_F1):.2f}  "
        f"VUS-ROC={pct(VUS_ROC):.2f}  VUS-PR={pct(VUS_PR):.2f}",
        flush=True,
    )

    if writer is not None:
        writer.add_scalar(f"{writer_prefix}/epoch_eval/AUC_PR",   A_PR,   ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/AUC_ROC",  A_ROC,  ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/F1",       F1,     ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/PA_F1",    PA_F1,  ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/Event_F1", EV_F1,  ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/R_F1",     R_F1,   ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/Aff_F",    Aff_F1, ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/VUS_ROC",  VUS_ROC,ep)
        writer.add_scalar(f"{writer_prefix}/epoch_eval/VUS_PR",   VUS_PR, ep)

    if csv_path is not None:
        row = {
            "epoch": ep, "is_final": int(ep == epochs_total),
            "AUC_PR": A_PR, "AUC_ROC": A_ROC,
            "F1": F1, "PA_F1": PA_F1, "Event_F1": EV_F1,
            "R_F1": R_F1, "Aff_F": Aff_F1,
            "VUS_ROC": VUS_ROC, "VUS_PR": VUS_PR,
        }
        write_header = not os.path.exists(csv_path)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{name}_seed{seed}_ep{ep}.pt")
        payload = {
            "epoch": ep,
            "seed": seed,
            "entity": name,
            "state_dict": model.state_dict(),
            "metrics": {
                "AUC_PR": A_PR, "AUC_ROC": A_ROC,
                "F1": F1, "PA_F1": PA_F1, "Event_F1": EV_F1,
                "R_F1": R_F1, "Aff_F": Aff_F1,
                "VUS_ROC": VUS_ROC, "VUS_PR": VUS_PR,
            },
            "cfg": cfg.dump(),
        }
        if mu is not None:
            payload["mu"] = np.asarray(mu, dtype=np.float32)
        if sd is not None:
            payload["sd"] = np.asarray(sd, dtype=np.float32)
        torch.save(payload, ckpt_path)
        print(f"  [ep {ep:02d} eval seed {seed}] saved checkpoint: {ckpt_path}", flush=True)

    if was_training:
        model.train()
