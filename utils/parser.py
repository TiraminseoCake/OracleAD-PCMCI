"""argparse + yacs config loader.

Usage:
    python main.py --cfg scripts/configs/swat.yaml \
        SEEDS "[0,1,2,3,4]" SOLVER.MAX_EPOCH 20
"""
import argparse
import os
import sys
import time

from config import get_cfg_defaults


# Dataset-specific defaults applied when the option is NOT set explicitly
# (via --cfg file or CLI opts). Mirrors the old behavior in picaad.py that
# picked a per-dataset LR fallback.
_DATASET_DEFAULT_LR = {
    'PSM': 5e-5,
    'SMD': 5e-4,
    'SWaT': 5e-4,
}


def parse_args():
    parser = argparse.ArgumentParser(description='PICAAD')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default=None,
                        help='Path to yacs YAML config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Overrides as "KEY VALUE KEY VALUE ..." pairs')
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def _build_auto_tag(cfg):
    """Derive an experiment tag from key config knobs."""
    parts = [cfg.MODEL.NAME.lower(), cfg.DATA.NAME.lower(), cfg.PICAAD.PRIOR.TYPE]
    parts.append(f'L{cfg.PICAAD.L}t{cfg.PICAAD.TAU_MAX}')
    parts.append(f'b{cfg.TRAIN.BATCH_SIZE}')
    parts.append(f'lr{cfg.SOLVER.BASE_LR:g}')
    return '_'.join(parts)


def load_config(args, date=None):
    """Compose the effective cfg from defaults + yaml + CLI opts.

    Returns (cfg, date). `date` is a shared timestamp used to derive run
    directories across seeds within a single invocation.
    """
    cfg = get_cfg_defaults()

    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    if args.opts:
        cfg.merge_from_list(args.opts)

    # Track which keys the caller set explicitly, so we don't clobber them
    # when applying dataset-specific fallbacks below.
    cli_keys = set(args.opts[0::2]) if args.opts else set()
    user_set_lr = 'SOLVER.BASE_LR' in cli_keys

    # Dataset-specific fallback for LR (mirrors old behavior).
    if not user_set_lr and cfg.DATA.NAME in _DATASET_DEFAULT_LR:
        cfg.SOLVER.BASE_LR = _DATASET_DEFAULT_LR[cfg.DATA.NAME]

    # Default INPUT_DIR to {BASE_DIR}/{NAME}_npz if empty
    if not cfg.DATA.INPUT_DIR:
        cfg.DATA.INPUT_DIR = os.path.join(cfg.DATA.BASE_DIR, f'{cfg.DATA.NAME}_npz')

    # Build run dir from tag + timestamp
    tag = str(cfg.EXP_TAG).strip() or _build_auto_tag(cfg)
    if date is None:
        date = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    run_name = f'{tag}_{date}'

    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR, cfg.DATA.NAME, run_name)
    if not cfg.TRAIN.CKPT_DIR:
        cfg.TRAIN.CKPT_DIR = os.path.join(cfg.RESULT_DIR, 'ckpt')

    return cfg, date
