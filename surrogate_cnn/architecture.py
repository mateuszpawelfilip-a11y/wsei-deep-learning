import json
import re

from .config import ALLOWED_TYPES, INPUT_SIZE, MAX_CONV_LAYERS, MAX_PARAMS
from .layers import normalize_layers


def _spatial_pass(layers: list[dict]) -> tuple[bool, str]:
    h, w = INPUT_SIZE, INPUT_SIZE
    channels = 3
    conv_count = 0
    has_gap = False
    linear_after_gap = False

    for layer in layers:
        t = layer["type"]
        if t == "conv":
            conv_count += 1
            if conv_count > MAX_CONV_LAYERS:
                return False, "Za dużo warstw Conv (max 6)"
            k = int(layer.get("kernel", 3))
            if k not in (1, 3, 5):
                return False, "Nieobsługiwany rozmiar kernela"
            pad = k // 2
            h = h - k + 1 + 2 * pad
            w = w - k + 1 + 2 * pad
            if h < 1 or w < 1:
                return False, "Wymiary przestrzenne spadły poniżej 1"
            channels = int(layer.get("filters", 32))
        elif t == "maxpool":
            h //= 2
            w //= 2
            if h < 1 or w < 1:
                return False, "Wymiary po MaxPool < 1"
        elif t == "batchnorm":
            pass
        elif t == "relu":
            pass
        elif t == "dropout":
            pass
        elif t == "globalaveragepooling":
            has_gap = True
            h, w = 1, 1
        elif t == "linear":
            if not has_gap and h * w > 1:
                return False, "Linear wymaga GAP lub spłaszczenia po redukcji wymiarów"
            linear_after_gap = True
        else:
            return False, f"Nieznany typ warstwy: {t}"

    if not linear_after_gap:
        return False, "Brak warstwy Linear (klasyfikator)"
    return True, "ok"


def validate_architecture(arch: dict) -> tuple[bool, str]:
    if "layers" not in arch or not arch["layers"]:
        return False, "Brak listy layers"
    layers = normalize_layers(arch["layers"])
    for layer in layers:
        if layer["type"] not in ALLOWED_TYPES:
            return False, f"Niedozwolony typ: {layer['type']}"
    ok, msg = _spatial_pass(layers)
    if not ok:
        return False, msg
    try:
        from .builder import build_model, count_parameters, smoke_test_forward

        model = build_model({"layers": layers})
        smoke_test_forward(model)
        n_params = count_parameters(model)
        if n_params > MAX_PARAMS:
            return False, f"Za dużo parametrów: {n_params} > {MAX_PARAMS}"
    except Exception as exc:
        return False, f"Błąd budowy modelu: {exc}"
    return True, "ok"


def parse_llm_json(text: str) -> dict | None:
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
