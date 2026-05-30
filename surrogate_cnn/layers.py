import copy

from .config import ALLOWED_TYPES


def normalize_layers(layers: list[dict]) -> list[dict]:
    out = []
    for layer in layers:
        t = layer.get("type", "").lower()
        if t in ("conv2d",):
            t = "conv"
        if t in ("maxpool2d",):
            t = "maxpool"
        if t in ("batchnorm2d",):
            t = "batchnorm"
        if t in ("fc",):
            t = "linear"
        if t in ("gap",):
            t = "globalaveragepooling"
        item = copy.deepcopy(layer)
        item["type"] = t
        out.append(item)
    return out
