"""PICAAD entry point.

Usage:
    python main.py --cfg scripts/configs/swat.yaml \
        SEEDS "[0,1,2,3,4]" SOLVER.MAX_EPOCH 20
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from datasets.build import list_entities, load_entity
from model.build import (
    apply_prior_to_model,
    build_causal_prior_cached,
    build_model,
)
from model.scoring import (
    fit_score_calibrator,
    score_components_to_timeline,
    score_windows,
)
from trainer import PicaadTrainer
from utils.analysis import (
    compute_intervention_contribution_3d,
    print_intervention_contrib_summary,
    save_intervention_contrib_csv,
)
from utils.evaluation import paper_eval_one
from utils.misc import mkdir, pct, safe_mean_std, set_devices, set_seed
from utils.parser import load_config, parse_args
from utils.tb_logging import SummaryWriter, TB_BACKEND


def _make_writer(cfg, entity_name, seed):
    if not cfg.TENSORBOARD.ENABLE or SummaryWriter is None:
        return None
    log_dir = os.path.join(cfg.TENSORBOARD.ROOT, entity_name, f'seed{seed}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    if hasattr(writer, 'add_text'):
        writer.add_text('config/backend', str(TB_BACKEND), 0)
        writer.add_text('config/entity',  entity_name, 0)
        writer.add_text('config/prior',   cfg.PICAAD.PRIOR.TYPE, 0)
    return writer


def _run_seed(cfg, entity, seed, te_weight_np, te_gate_np, device):
    """Train + evaluate one (entity, seed) run. Returns metric tuple."""
    print(f'\n[seed {seed}] training ...', flush=True)
    set_seed(seed)

    writer = _make_writer(cfg, entity.name, seed)

    model = build_model(cfg, N=entity.N).to(device)
    apply_prior_to_model(cfg, model, te_weight_np, te_gate_np)

    ckpt_dir = cfg.TRAIN.CKPT_DIR
    csv_path = os.path.join(cfg.RESULT_DIR, f'{entity.name}_seed{seed}_epoch_metrics.csv')

    trainer = PicaadTrainer(
        cfg, model, entity, seed,
        device=device, writer=writer,
        ckpt_dir=ckpt_dir, csv_path=csv_path,
    )
    trainer.train()

    return _final_eval(cfg, model, entity, seed, writer, device)


def _final_eval(cfg, model, entity, seed, writer, device):
    """Final-model eval (after last epoch) with optional diagnostics, mask
    contribution analysis, and per-seed .npz dump."""
    calibrator = None
    if cfg.PICAAD.SCORING.CALIBRATE:
        print(f'[seed {seed}] fitting robust score calibrator on train windows ...', flush=True)
        train_scores = score_windows(model, entity.train_z, device,
                                     batch=cfg.TRAIN.BATCH_SIZE,
                                     scoring_cfg=cfg.PICAAD.SCORING,
                                     calibrator=None)
        calibrator = fit_score_calibrator(train_scores)

    test_scores = score_windows(model, entity.test_z, device,
                                batch=cfg.TEST.BATCH_SIZE,
                                scoring_cfg=cfg.PICAAD.SCORING,
                                calibrator=calibrator)

    Tt = entity.test_z.shape[0]
    start = cfg.PICAAD.L - 1
    score_t_dict = score_components_to_timeline(
        {k: test_scores[k] for k in ['P', 'C', 'G', 'S', 'A', 'P_raw', 'C_raw', 'G_raw']},
        Tt=Tt, start=start,
    )
    P_t = score_t_dict['P_t']; C_t = score_t_dict['C_t']; G_t = score_t_dict['G_t']
    S_t = score_t_dict['S_t']; A_t = score_t_dict['A_t']

    mtr_P = paper_eval_one(P_t, entity.y, start, cfg.EVAL)
    mtr_C = paper_eval_one(C_t, entity.y, start, cfg.EVAL)
    mtr_G = paper_eval_one(G_t, entity.y, start, cfg.EVAL)
    mtr_S = paper_eval_one(S_t, entity.y, start, cfg.EVAL)
    mtr_A = paper_eval_one(A_t, entity.y, start, cfg.EVAL)

    if cfg.EVAL.DIAGNOSE_COMPONENTS:
        print(
            f'[seed {seed}] paper_eval components\n'
            f"  P-only : A-PR={pct(mtr_P['AUC-PR']):.2f}  VUS-PR={pct(mtr_P['VUS-PR']):.2f}  F1={pct(mtr_P['Standard-F1']):.2f}\n"
            f"  C-only : A-PR={pct(mtr_C['AUC-PR']):.2f}  VUS-PR={pct(mtr_C['VUS-PR']):.2f}  F1={pct(mtr_C['Standard-F1']):.2f}\n"
            f"  G-only : A-PR={pct(mtr_G['AUC-PR']):.2f}  VUS-PR={pct(mtr_G['VUS-PR']):.2f}  F1={pct(mtr_G['Standard-F1']):.2f}\n"
            f"  S=C+G  : A-PR={pct(mtr_S['AUC-PR']):.2f}  VUS-PR={pct(mtr_S['VUS-PR']):.2f}  F1={pct(mtr_S['Standard-F1']):.2f}\n"
            f"  A=P*S  : A-PR={pct(mtr_A['AUC-PR']):.2f}  VUS-PR={pct(mtr_A['VUS-PR']):.2f}  F1={pct(mtr_A['Standard-F1']):.2f}",
            flush=True,
        )

    A_PR = float(mtr_A['AUC-PR']); A_ROC = float(mtr_A['AUC-ROC'])
    VUS_PR = float(mtr_A['VUS-PR']); VUS_ROC = float(mtr_A['VUS-ROC'])
    F1 = float(mtr_A['Standard-F1']); PA_F1 = float(mtr_A['PA-F1'])
    EV_F1 = float(mtr_A['Event-based-F1']); R_F1 = float(mtr_A['R-based-F1'])
    Aff_F1 = float(mtr_A['Affiliation-F'])

    mask_out = None
    if cfg.ANALYSIS.MASK_CONTRIB:
        print(f'[seed {seed}] running local intervention contribution analysis ...', flush=True)
        batch = cfg.ANALYSIS.MASK_BATCH if cfg.ANALYSIS.MASK_BATCH > 0 else cfg.TEST.BATCH_SIZE
        mask_out = compute_intervention_contribution_3d(
            model, entity.test_z, device, batch=batch,
            mode=cfg.PICAAD.INTERVENTION.MODE,
            fill_value=cfg.PICAAD.INTERVENTION.FILL_VALUE,
        )
        print_intervention_contrib_summary(entity.name, mask_out['G_pos_tau'], topk=cfg.ANALYSIS.MASK_TOPK)

    print(
        f'[seed {seed}] '
        f'A-PR={pct(A_PR):.2f}  A-ROC={pct(A_ROC):.2f}  '
        f'F1={pct(F1):.2f}  PA-F1={pct(PA_F1):.2f}  EV-F1={pct(EV_F1):.2f}  '
        f'R-F1={pct(R_F1):.2f}  Aff-F={pct(Aff_F1):.2f}  '
        f'VUS-ROC={pct(VUS_ROC):.2f}  VUS-PR={pct(VUS_PR):.2f}',
        flush=True,
    )

    if writer is not None:
        for name, val in [('AUC_PR', A_PR), ('AUC_ROC', A_ROC), ('F1', F1),
                          ('PA_F1', PA_F1), ('Event_F1', EV_F1), ('R_F1', R_F1),
                          ('Aff_F', Aff_F1), ('VUS_ROC', VUS_ROC), ('VUS_PR', VUS_PR)]:
            writer.add_scalar(f'{entity.name}/eval/{name}', val, seed)
        writer.flush(); writer.close()

    if cfg.SAVE.PER_SEED:
        save_kwargs = {
            'A_t': A_t, 'S_t': S_t, 'P_t': P_t, 'C_t': C_t, 'G_t': G_t,
            'P_raw_t': score_t_dict['P_raw_t'],
            'C_raw_t': score_t_dict['C_raw_t'],
            'G_raw_t': score_t_dict['G_raw_t'],
            'y': entity.y,
            'cls_ref':             model.cls_ref.detach().cpu().numpy().astype(np.float32),
            'w_ref':               model.w_ref.detach().cpu().numpy().astype(np.float32),
            'gate':                model.edge_gate().detach().cpu().numpy().astype(np.float32),
            'pred_weights_global': model.get_pred_weights().detach().cpu().numpy().astype(np.float32),
            'te_prior_weight':     model.te_prior_weight.detach().cpu().numpy().astype(np.float32),
            'te_prior_gate':       model.te_prior_gate.detach().cpu().numpy().astype(np.float32),
            'mu': entity.mu, 'sd': entity.sd,
        }
        if calibrator is not None:
            for comp in ['P', 'C', 'G']:
                save_kwargs[f'{comp}_center'] = np.array([calibrator[f'{comp}_raw']['center']], dtype=np.float32)
                save_kwargs[f'{comp}_scale']  = np.array([calibrator[f'{comp}_raw']['scale']],  dtype=np.float32)
        if mask_out is not None:
            save_kwargs.update({k: mask_out[k] for k in mask_out})
        np.savez(os.path.join(cfg.RESULT_DIR, f'{entity.name}_seed{seed}.npz'), **save_kwargs)

        if cfg.ANALYSIS.MASK_SAVE_CSV and mask_out is not None:
            save_intervention_contrib_csv(
                os.path.join(cfg.RESULT_DIR, f'{entity.name}_seed{seed}_intervention_contrib.csv'),
                mask_out['G_raw_tau'], mask_out['G_pos_tau'],
            )

    return (A_PR, A_ROC, F1, PA_F1, EV_F1, R_F1, Aff_F1, VUS_ROC, VUS_PR)


def _summarize_and_save(cfg, per_entity_rows):
    """Write RESULT_DIR/summary.csv (one row per entity, seed-mean values)."""
    if not per_entity_rows:
        return
    df = pd.DataFrame(per_entity_rows, columns=[
        'entity', 'AUC_PR', 'AUC_ROC',
        'F1', 'PA_F1', 'Event_F1', 'R_F1', 'Aff_F',
        'VUS_ROC', 'VUS_PR',
    ])
    out_path = os.path.join(cfg.RESULT_DIR, 'summary.csv')
    df.to_csv(out_path, index=False)
    print(f'\nSaved summary: {out_path}', flush=True)


def main():
    args = parse_args()
    cfg, _ = load_config(args)

    # Only apply cfg.VISIBLE_DEVICES when the env is not already restricted by
    # a caller (e.g., scripts/run_parallel.py sets CUDA_VISIBLE_DEVICES per
    # subprocess for GPU sharding). Overriding it here would clobber the
    # per-subprocess assignment and force every job onto GPU 0.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        set_devices(cfg.VISIBLE_DEVICES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")})',
          flush=True)

    mkdir(cfg.RESULT_DIR)
    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    print(f'Saved run config: {os.path.join(cfg.RESULT_DIR, "config.yaml")}', flush=True)

    entities = list_entities(cfg)
    if not entities:
        print(f'[error] no entities found under {cfg.DATA.INPUT_DIR}', flush=True)
        sys.exit(1)

    seeds = list(cfg.SEEDS)
    per_entity_rows = []

    for name in entities:
        try:
            entity = load_entity(cfg, name)
        except FileNotFoundError as e:
            print(f'[skip] {e}', flush=True)
            continue
        except ValueError as e:
            print(f'[skip] {name}: {e}', flush=True)
            continue

        if (entity.T_train < cfg.PICAAD.L + 1) or (entity.T_test < cfg.PICAAD.L + 1):
            print(f'[skip] {name} too short (Ttr={entity.T_train}, Tte={entity.T_test})',
                  flush=True)
            continue

        prior_label = cfg.PICAAD.PRIOR.TYPE.upper()
        print(f'\n=== {name} (Ttr={entity.T_train}, Tte={entity.T_test}, N={entity.N}) '
              f'| prior={prior_label} ===', flush=True)

        te_weight_np, te_gate_np = build_causal_prior_cached(cfg, entity.train_z, entity.name)

        print(
            f'lr={cfg.SOLVER.BASE_LR} L={cfg.PICAAD.L} tau_max={cfg.PICAAD.TAU_MAX} '
            f'lag_win={cfg.PICAAD.LAG_WIN} batch={cfg.TRAIN.BATCH_SIZE} '
            f'enc/dec={cfg.PICAAD.ENC_LAYERS}/{cfg.PICAAD.DEC_LAYERS} '
            f'lam_task={cfg.PICAAD.LAM_TASK} lam_causal={cfg.PICAAD.LAM_CAUSAL} '
            f'lam_graphreg={cfg.PICAAD.LAM_GRAPHREG} lam_robust={cfg.PICAAD.LAM_ROBUST} '
            f'prior={prior_label} te_prior_blend={cfg.PICAAD.PRIOR.BLEND} '
            f'te_init_scale={cfg.PICAAD.PRIOR.INIT_SCALE}',
            flush=True,
        )

        metrics = []
        for seed in seeds:
            metrics.append(_run_seed(cfg, entity, seed, te_weight_np, te_gate_np, device))

        A_PR_m,   A_PR_s   = safe_mean_std([m[0] for m in metrics])
        A_ROC_m,  A_ROC_s  = safe_mean_std([m[1] for m in metrics])
        F1_m,     F1_s     = safe_mean_std([m[2] for m in metrics])
        PA_m,     PA_s     = safe_mean_std([m[3] for m in metrics])
        EV_m,     EV_s     = safe_mean_std([m[4] for m in metrics])
        R_F1_m,   R_F1_s   = safe_mean_std([m[5] for m in metrics])
        Aff_m,    Aff_s    = safe_mean_std([m[6] for m in metrics])
        VUS_ROC_m,VUS_ROC_s= safe_mean_std([m[7] for m in metrics])
        VUS_PR_m, VUS_PR_s = safe_mean_std([m[8] for m in metrics])

        print(f'\n[{name}] mean+/-std over {len(seeds)} seeds:', flush=True)
        for label, m, s in [
            ('A-PR', A_PR_m, A_PR_s), ('A-ROC', A_ROC_m, A_ROC_s),
            ('Standard-F1', F1_m, F1_s), ('PA-F1', PA_m, PA_s),
            ('Event-F1', EV_m, EV_s), ('R-based-F1', R_F1_m, R_F1_s),
            ('Affiliation-F', Aff_m, Aff_s),
            ('VUS-ROC', VUS_ROC_m, VUS_ROC_s), ('VUS-PR', VUS_PR_m, VUS_PR_s),
        ]:
            print(f'  {label:14s} {pct(m):.2f} +/- {pct(s):.2f}', flush=True)

        per_entity_rows.append(
            (name, A_PR_m, A_ROC_m, F1_m, PA_m, EV_m, R_F1_m, Aff_m, VUS_ROC_m, VUS_PR_m)
        )

    _summarize_and_save(cfg, per_entity_rows)


if __name__ == '__main__':
    main()
