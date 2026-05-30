"""Surrogate-based CNN accuracy prediction for CIFAR-10."""

from .config import FAST_MODE, PROJECT_DIR
from .encoding import encode_architecture, decode_vector_preview
from .architecture import validate_architecture
from .layers import normalize_layers
from .builder import build_model, count_parameters
from .trainer import train_architecture, get_cifar_loaders
from .llm_generator import generate_architecture_llm, generate_architecture_random
from .surrogates import SurrogateMLP, SurrogateXGB, train_surrogates, predict_surrogates
from .metrics import evaluate_surrogates
from .pipeline import run_initial_collection, run_iterative_search, load_dataset, save_dataset

__all__ = [
    "FAST_MODE",
    "PROJECT_DIR",
    "encode_architecture",
    "decode_vector_preview",
    "validate_architecture",
    "normalize_layers",
    "build_model",
    "count_parameters",
    "train_architecture",
    "get_cifar_loaders",
    "generate_architecture_llm",
    "generate_architecture_random",
    "SurrogateMLP",
    "SurrogateXGB",
    "train_surrogates",
    "predict_surrogates",
    "evaluate_surrogates",
    "run_initial_collection",
    "run_iterative_search",
    "load_dataset",
    "save_dataset",
]
