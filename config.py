from yacs.config import CfgNode as CN

_C = CN()

# --- global ---
_C.SEED = 0
_C.SEEDS = [0, 1, 2, 3, 4]
_C.VISIBLE_DEVICES = 0
_C.RESULT_DIR = 'results/'
_C.EXP_TAG = ''  # empty -> auto tag


# --- data ---
_C.DATA = CN()
_C.DATA.BASE_DIR = 'data/'          # symlink to shared data root
_C.DATA.NAME = 'SWaT'               # dataset name (used in path composition)
_C.DATA.INPUT_DIR = ''              # NPZ dir; empty -> {BASE_DIR}/{NAME}_npz
_C.DATA.ENTITIES = ''               # comma-separated entities; empty -> glob
_C.DATA.SCALE = 'standard'          # standard | none
_C.DATA.NUM_ENVS = 4                # pseudo-env count for invariance loss


# --- data loader ---
_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_WORKERS = 0
_C.DATA_LOADER.PIN_MEMORY = False
_C.DATA_LOADER.DROP_LAST = False


# --- train / test ---
_C.TRAIN = CN()
_C.TRAIN.ENABLE = True
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = False
_C.TRAIN.EVAL_EVERY = 5             # eval + ckpt every N epochs (0 disables)
_C.TRAIN.CKPT_DIR = ''              # empty -> {RESULT_DIR}/ckpt

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128


# --- solver ---
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 80
_C.SOLVER.OPTIMIZING_METHOD = 'adamw'
_C.SOLVER.BASE_LR = 5e-4
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.GRADIENT_CLIP = 1.0       # 0 disables


# --- model dispatch ---
_C.MODEL = CN()
_C.MODEL.NAME = 'PICAAD'


# --- PICAAD ---
_C.PICAAD = CN()

# architecture
_C.PICAAD.L = 10                     # window length
_C.PICAAD.TAU_MAX = 5
_C.PICAAD.LAG_WIN = 5
_C.PICAAD.D = 64
_C.PICAAD.HEADS = 4
_C.PICAAD.DROPOUT = 0.0
_C.PICAAD.ENC_LAYERS = 2
_C.PICAAD.DEC_LAYERS = 2
_C.PICAAD.MHSA_RESIDUAL = False
_C.PICAAD.LAG_FUSION = 'mean'
_C.PICAAD.PRED_TEMP = 1.0
_C.PICAAD.SELF_LOOP_BIAS = 1.0
_C.PICAAD.LAG_SOURCE_TOPK = 0
_C.PICAAD.DYNAMIC_GRAPH = True
_C.PICAAD.GRAPH_HIDDEN = 16
_C.PICAAD.GATE_INIT = 0.15
_C.PICAAD.CAUSAL_ATTN_MASK_SCALE = 0.5
_C.PICAAD.CAUSAL_MASK_WARMUP = 5

# loss weights
_C.PICAAD.LAM_TASK = 1.0
_C.PICAAD.LAM_CAUSAL = 1.0
_C.PICAAD.LAM_GRAPHREG = 0.05
_C.PICAAD.LAM_ROBUST = 0.10
_C.PICAAD.CLS_EMA = 0.9
_C.PICAAD.WREF_EMA = 0.9
_C.PICAAD.START_CLS_EPOCH = 5
_C.PICAAD.START_WREF_EPOCH = 3
_C.PICAAD.TRAIN_LOSS_TYPE = 'l1'     # l1 | l2root
_C.PICAAD.RECON_LOSS_TYPE = 'l1'

# causal prior
_C.PICAAD.PRIOR = CN()
_C.PICAAD.PRIOR.TYPE = 'pcmci'       # te | cte | pcmci
_C.PICAAD.PRIOR.BLEND = 0.35         # te_prior_blend
_C.PICAAD.PRIOR.INIT_SCALE = 0.25    # te_init_scale
_C.PICAAD.PRIOR.SELF_MASS = 0.25
_C.PICAAD.PRIOR.SEED = 0
# On-disk prior cache: skip rebuild when hyperparams+data hash unchanged
_C.PICAAD.PRIOR.CACHE_DIR = 'data/prior_cache'
_C.PICAAD.PRIOR.CACHE_ENABLE = True
_C.PICAAD.PRIOR.CACHE_REBUILD = False

_C.PICAAD.PRIOR.PCMCI = CN()
_C.PICAAD.PRIOR.PCMCI.CI_TEST = 'ParCorr'
_C.PICAAD.PRIOR.PCMCI.ALPHA = 0.05
_C.PICAAD.PRIOR.PCMCI.SUBSAMPLE = 10000

_C.PICAAD.PRIOR.TE = CN()
_C.PICAAD.PRIOR.TE.BINS = 8
_C.PICAAD.PRIOR.TE.NUM_CHUNKS = 32
_C.PICAAD.PRIOR.TE.CHUNK_LEN = 256
_C.PICAAD.PRIOR.TE.THRESHOLD = 0.0

# intervention (train-time perm-align + post-hoc analysis share flags)
_C.PICAAD.INTERVENTION = CN()
_C.PICAAD.INTERVENTION.PERM_PAIRS_PER_BATCH = 2
_C.PICAAD.INTERVENTION.PERM_MODE = 'permute'   # permute | fill
_C.PICAAD.INTERVENTION.MODE = 'permute'        # for post-hoc analysis
_C.PICAAD.INTERVENTION.FILL_VALUE = 0.0

# scoring
_C.PICAAD.SCORING = CN()
_C.PICAAD.SCORING.P_AGG = 'mean'     # mean | max | topk
_C.PICAAD.SCORING.P_TOPK = 3
_C.PICAAD.SCORING.C_AGG = 'fro'      # fro | maxrow | topkrow
_C.PICAAD.SCORING.C_TOPK = 3
_C.PICAAD.SCORING.G_AGG = 'fro'
_C.PICAAD.SCORING.G_TOPK = 3
_C.PICAAD.SCORING.CAUSAL_LAG_AGG = 'mean'
_C.PICAAD.SCORING.GRAPH_LAG_AGG = 'mean'
_C.PICAAD.SCORING.SCORE_ALPHA = 1.0
_C.PICAAD.SCORING.SCORE_BETA = 1.0
_C.PICAAD.SCORING.CALIB_CLIP_MIN = 0.0
_C.PICAAD.SCORING.CALIBRATE = False


# --- paper eval ---
_C.EVAL = CN()
_C.EVAL.SLIDING_WINDOW = 100
_C.EVAL.VUS_VERSION = 'opt'
_C.EVAL.VUS_THRE = 250
_C.EVAL.USE_MEDIAN_VUS_WINDOW = False
_C.EVAL.DIAGNOSE_COMPONENTS = False


# --- analysis (intervention contribution) ---
_C.ANALYSIS = CN()
_C.ANALYSIS.MASK_CONTRIB = False
_C.ANALYSIS.MASK_BATCH = 0
_C.ANALYSIS.MASK_TOPK = 10
_C.ANALYSIS.MASK_SAVE_CSV = False


# --- tensorboard ---
_C.TENSORBOARD = CN()
_C.TENSORBOARD.ENABLE = False
_C.TENSORBOARD.ROOT = 'runs/tensorboard/picaad'
_C.TENSORBOARD.HISTOGRAMS = False
_C.TENSORBOARD.FIGURES = False


# --- save ---
_C.SAVE = CN()
_C.SAVE.PER_SEED = False


def get_cfg_defaults():
    return _C.clone()
