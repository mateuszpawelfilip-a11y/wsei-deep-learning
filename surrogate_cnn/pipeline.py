import json
import time
from pathlib import Path

import numpy as np
import torch

from .architecture import validate_architecture
from .builder import build_model
from .config import (
    ARCHIVE_PATH,
    DATA_DIR,
    FAST_MODE,
    INITIAL_ARCH_COUNT,
    ITERATIONS,
    TRAIN_EPOCHS,
)
from .encoding import encode_architecture
from .llm_generator import generate_architecture
from .metrics import evaluate_surrogates, print_metrics_table
from .surrogates import predict_surrogates, train_surrogates
from .trainer import train_architecture


def load_dataset(path: Path = ARCHIVE_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_dataset(records: list[dict], path: Path = ARCHIVE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _train_and_record(arch: dict, device: torch.device, source: str) -> dict | None:
    ok, msg = validate_architecture(arch)
    if not ok:
        print(f"  Pominięto (niepoprawna): {msg}")
        return None
    model = build_model(arch)
    acc = train_architecture(model, device, epochs=TRAIN_EPOCHS)
    vec = encode_architecture(arch)
    return {
        "architecture": arch,
        "accuracy": float(acc),
        "accuracy_pct": round(acc * 100, 2),
        "vector": vec.tolist(),
        "source": source,
    }


def run_initial_collection(
    n: int = INITIAL_ARCH_COUNT,
    device: torch.device | None = None,
    use_llm: bool = True,
) -> list[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_dataset()
    existing = len(records)
    target = max(0, n - existing)
    if target == 0:
        print(f"Dataset już ma {existing} architektur – ładuję z pliku.")
        return records

    print(f"Generowanie i trening {target} architektur (device={device})...")
    for i in range(target):
        arch, src = generate_architecture(use_llm=use_llm)
        print(f"[{i + 1}/{target}] Źródło: {src}")
        rec = _train_and_record(arch, device, src)
        if rec:
            records.append(rec)
            save_dataset(records)
            print(f"  Accuracy: {rec['accuracy_pct']}%")
    return records


def _matrix_from_records(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([r["vector"] for r in records], dtype=np.float32)
    y = np.array([r["accuracy"] for r in records], dtype=np.float32)
    return X, y


def evaluate_all_surrogates(records: list[dict], device: torch.device) -> list[dict]:
    X, y = _matrix_from_records(records)
    mlp, xgb, scaler = train_surrogates(X, y, device, mlp_epochs=80 if FAST_MODE else 200)

    mlp_preds, xgb_preds = [], []
    for i in range(len(X)):
        preds = predict_surrogates(mlp, xgb, scaler, X[i], device)
        mlp_preds.append(preds["mlp"])
        xgb_preds.append(preds["xgb"])

    rows = [
        evaluate_surrogates(y, np.array(mlp_preds), "MLP"),
        evaluate_surrogates(y, np.array(xgb_preds), "XGBoost"),
    ]
    print_metrics_table(rows)
    return rows


def run_iterative_search(
    iterations: int = ITERATIONS,
    device: torch.device | None = None,
    use_llm: bool = True,
) -> list[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_dataset()
    if len(records) < 5:
        records = run_initial_collection(device=device, use_llm=use_llm)

    X, y = _matrix_from_records(records)
    mlp, xgb, scaler = train_surrogates(X, y, device, mlp_epochs=80 if FAST_MODE else 200)
    best_acc = float(y.max())
    print(f"Start: {len(records)} architektur, najlepsze Accuracy = {best_acc * 100:.2f}%")

    for it in range(1, iterations + 1):
        arch, src = generate_architecture(use_llm=use_llm)
        vec = encode_architecture(arch)
        preds = predict_surrogates(mlp, xgb, scaler, vec, device)
        pred_acc = preds["ensemble"]
        print(
            f"\nIteracja {it}/{iterations} | źródło={src} | "
            f"pred MLP={preds['mlp']*100:.1f}% XGB={preds['xgb']*100:.1f}%"
        )

        if pred_acc <= best_acc:
            print(f"  Pominięto trening (pred {pred_acc*100:.2f}% <= best {best_acc*100:.2f}%)")
            continue

        print(f"  Predykcja lepsza – pełny trening CNN...")
        rec = _train_and_record(arch, device, f"{src}_iter{it}")
        if rec is None:
            continue
        records.append(rec)
        save_dataset(records)
        true_acc = rec["accuracy"]
        print(f"  Rzeczywiste Accuracy: {rec['accuracy_pct']}%")
        best_acc = max(best_acc, true_acc)

        X, y = _matrix_from_records(records)
        mlp, xgb, scaler = train_surrogates(X, y, device, mlp_epochs=60 if FAST_MODE else 150)
        time.sleep(0.1)

    print(f"\nKoniec iteracji. Dataset: {len(records)} architektur, best={best_acc*100:.2f}%")
    return records
