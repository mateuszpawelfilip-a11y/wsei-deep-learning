from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "surrogate_cnn"
ARCHIVE_PATH = DATA_DIR / "architectures.json"

# Szybki tryb do testów w notebooku (pełny: 60 arch, 15 epok)
FAST_MODE = False

NUM_CLASSES = 10
INPUT_SIZE = 32
MAX_CONV_LAYERS = 6
MAX_PARAMS = 3_000_000
MAX_LAYER_SLOTS = 15
FEATURES_PER_SLOT = 3
VECTOR_DIM = MAX_LAYER_SLOTS * FEATURES_PER_SLOT

INITIAL_ARCH_COUNT = 2 if FAST_MODE else 60
TRAIN_EPOCHS = 3 if FAST_MODE else 15
BATCH_SIZE = 128
ITERATIONS = 5 if FAST_MODE else 20

# Hugging Face – generator architektur (AutoModelForCausalLM.from_pretrained)
HF_MODEL_NAME = "Qwen/Qwen3.5-2B" 
HF_MAX_NEW_TOKENS = 384
HF_TEMPERATURE = 0.8

LAYER_CODES = {
    "conv": 1,
    "conv2d": 1,
    "maxpool": 2,
    "maxpool2d": 2,
    "dropout": 3,
    "linear": 4,
    "fc": 4,
    "batchnorm": 5,
    "batchnorm2d": 5,
    "relu": 6,
    "globalaveragepooling": 7,
    "gap": 7,
}

CODE_TO_NAME = {
    1: "Conv",
    2: "MaxPool",
    3: "Dropout",
    4: "Linear",
    5: "BatchNorm",
    6: "ReLU",
    7: "GAP",
}

ALLOWED_TYPES = {
    "conv",
    "conv2d",
    "batchnorm",
    "batchnorm2d",
    "relu",
    "maxpool",
    "maxpool2d",
    "dropout",
    "linear",
    "fc",
    "globalaveragepooling",
    "gap",
}
