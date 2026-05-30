from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from .config import MUSIC_DURATION_S, MUSIC_MAX_NEW_TOKENS, MUSIC_MODELS


_models: dict[str, MusicgenForConditionalGeneration] = {}
_processors: dict[str, AutoProcessor] = {}


def _sampling_rate(model: MusicgenForConditionalGeneration) -> int:
    return int(model.config.audio_encoder.sampling_rate)


def get_musicgen(model_key: str, device: str):
    if model_key not in MUSIC_MODELS:
        raise KeyError(f"Unknown model_key: {model_key}")
    if model_key not in _models:
        model_id = MUSIC_MODELS[model_key]
        proc = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)
        model.eval()
        _processors[model_key] = proc
        _models[model_key] = model
    return _processors[model_key], _models[model_key]


def generate_music(
    prompt: str,
    model_key: str = "musicgen_small",
    device: str = "cuda",
) -> tuple[np.ndarray, int]:
    processor, model = get_musicgen(model_key, device)
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=MUSIC_MAX_NEW_TOKENS, do_sample=True)

    wav = audio[0, 0].float().cpu().numpy()
    sr = _sampling_rate(model)
    # przybliżone przycięcie do MUSIC_DURATION_S
    max_samples = int(MUSIC_DURATION_S * sr)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    return wav.astype(np.float32), int(sr)


def unload_music_models() -> None:
    global _models, _processors
    _models.clear()
    _processors.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
