import numpy as np

from .config import (
    CODE_TO_NAME,
    FEATURES_PER_SLOT,
    LAYER_CODES,
    MAX_LAYER_SLOTS,
    VECTOR_DIM,
)


def _layer_triplet(layer: dict) -> tuple[float, float, float]:
    t = layer.get("type", "").lower().replace("2d", "")
    if t in ("conv",):
        return (
            float(LAYER_CODES["conv"]),
            float(layer.get("filters", 32)),
            float(layer.get("kernel", 3)),
        )
    if t in ("maxpool",):
        return (float(LAYER_CODES["maxpool"]), 0.0, 0.0)
    if t == "dropout":
        return (float(LAYER_CODES["dropout"]), float(layer.get("p", 0.2)), 0.0)
    if t in ("linear", "fc"):
        return (float(LAYER_CODES["linear"]), float(layer.get("units", 128)), 0.0)
    if t in ("batchnorm",):
        return (float(LAYER_CODES["batchnorm"]), 0.0, 0.0)
    if t == "relu":
        return (float(LAYER_CODES["relu"]), 0.0, 0.0)
    if t in ("globalaveragepooling", "gap"):
        return (float(LAYER_CODES["globalaveragepooling"]), 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def encode_architecture(arch: dict) -> np.ndarray:
    """Fixed-length vector: każda warstwa → [type, param1, param2], padding zerami."""
    layers = arch.get("layers", [])
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    for i, layer in enumerate(layers[:MAX_LAYER_SLOTS]):
        a, b, c = _layer_triplet(layer)
        base = i * FEATURES_PER_SLOT
        vec[base] = a
        vec[base + 1] = b
        vec[base + 2] = c
    return vec


def decode_vector_preview(vec: np.ndarray) -> list[str]:
    """Czytelny podgląd zakodowanego wektora."""
    out = []
    for i in range(MAX_LAYER_SLOTS):
        base = i * FEATURES_PER_SLOT
        code = int(vec[base])
        if code == 0:
            break
        name = CODE_TO_NAME.get(code, "?")
        p1, p2 = vec[base + 1], vec[base + 2]
        if code == 1:
            out.append(f"Conv({int(p1)},{int(p2)})")
        elif code == 3:
            out.append(f"Dropout({p1})")
        elif code == 4:
            out.append(f"Linear({int(p1)})")
        else:
            out.append(name)
    return out
