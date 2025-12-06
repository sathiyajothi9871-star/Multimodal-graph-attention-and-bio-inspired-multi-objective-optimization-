import os

# Root project directory (auto-detected from this file location)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- Data paths (adjust to your setup) ----
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed")

RAW_CT_DIR = os.path.join(RAW_DATA_DIR, "ct")
RAW_CLINICAL_CSV = os.path.join(RAW_DATA_DIR, "clinical.csv")

PROCESSED_CT_DIR = os.path.join(PROCESSED_DATA_DIR, "ct_npy")
PROCESSED_CLINICAL_CSV = os.path.join(PROCESSED_DATA_DIR, "clinical.csv")
SPLITS_JSON = os.path.join(PROCESSED_DATA_DIR, "splits.json")

# ---- Outputs ----
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUT_ROOT, "figures")
METRICS_DIR = os.path.join(OUTPUT_ROOT, "metrics")
ABLATION_DIR = os.path.join(OUTPUT_ROOT, "ablation")

for d in [OUTPUT_ROOT, CHECKPOINT_DIR, FIGURES_DIR, METRICS_DIR, ABLATION_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Model & training hyperparameters ----
SEED = 42
DEVICE = "cuda"  # or "cpu"

BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# For MO-GWO
USE_MO_GWO = True
MO_GWO_NUM_WOLVES = 8
MO_GWO_NUM_ITERS = 10

# 3D CNN input shape
CT_DEPTH = 128
CT_HEIGHT = 128
CT_WIDTH = 128

NUM_CLINICAL_FEATURES = 16   # adjust after preprocessing
NUM_RADIOMIC_FEATURES = 32   # adjust if you compute radiomics

# Graph
GAT_HIDDEN_DIM = 128
GAT_HEADS = 4
GRAPH_EMB_DIM = 128
IMG_EMB_DIM = 256
FUSION_EMB_DIM = 128
NUM_CLASSES = 2
