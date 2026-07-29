"""Hyperparameters. Edit here, not in train.py."""
from pathlib import Path

HERE = Path(__file__).parent
DATASET = HERE / "dataset.npz"
OUTPUT = HERE / "runs"
OUTPUT.mkdir(exist_ok=True)

W, H = 800.0, 500.0
MAX_N = 40
K_MAX = 20

D_MODEL = 128
N_BLOCKS = 4
N_RBF = 16
RBF_W = 0.08
SIGMAS = [0.06, 0.12, 0.30]
N_SIGMA = len(SIGMAS)
GRU_HIDDEN = 128
DROPOUT = 0.1

BATCH = 256
LR = 6e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 25
SADDLE_BY = 25
MIN_EPOCHS = 40
GRAD_CLIP = 1.0
WARMUP = 5

W_PAIR = 1.0
W_ENTROPY = 0.01
PAIR_BALANCE = False

W_REG_K = 0.0
W_REG_CENTROID = 0.0
W_REG_SIL = 0.0

SPLIT_BY = "base_uuid"
VAL_FRAC, TEST_FRAC, SPLIT_SEED = 0.10, 0.15, 42

DEVICE = "cuda"
MAX_HOURS = 3.0
JOBS = 4
