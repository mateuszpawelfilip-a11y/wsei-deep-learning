from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import signal

from .utils import load_audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from .config import AUDIO_CLASSIFIER, BLIP_MODEL

_blip_model = None
_blip_proc = None
_audio_model = None
_audio_extractor = None


def describe_image(img: Image.Image, device: str) -> str:
    global _blip_model, _blip_proc
    if _blip_model is None:
        _blip_proc = BlipProcessor.from_pretrained(BLIP_MODEL)
        _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

    inputs = _blip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _blip_model.generate(**inputs, max_new_tokens=60)
    return _blip_proc.decode(out[0], skip_special_tokens=True)


def _resample(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return waveform
    num_samples = int(round(len(waveform) * target_sr / sr))
    return signal.resample(waveform, num_samples).astype(np.float32)


def audio_to_caption(audio_path: Path, device: str) -> str:
    """Audio → etykiety (AST) → tekst pod Stable Diffusion (music-to-image proxy)."""
    global _audio_model, _audio_extractor
    if _audio_model is None:
        _audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_CLASSIFIER)
        _audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_CLASSIFIER).to(device)

    waveform, sr = load_audio(audio_path)
    target_sr = _audio_extractor.sampling_rate
    waveform = _resample(waveform, sr, target_sr)

    inputs = _audio_extractor(
        waveform,
        sampling_rate=target_sr,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = _audio_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    topk = torch.topk(probs, k=3)
    labels = [_audio_model.config.id2label[i.item()] for i in topk.indices]

    mood = ", ".join(labels)
    return (
        f"WSEI university campus in Krakow, IDEIS DSW, academic atmosphere, "
        f"mood inspired by music: {mood}, students, modern architecture, digital art"
    )


def compare_prompts(start: str, end_caption: str) -> dict:
    start_words = {w.lower() for w in start.replace(",", " ").split() if len(w) > 3}
    end_words = {w.lower() for w in end_caption.replace(",", " ").split() if len(w) > 3}
    common = sorted(start_words & end_words)
    return {
        "start_prompt": start,
        "end_caption": end_caption,
        "common_keywords": common,
        "start_only": sorted(start_words - end_words)[:15],
        "end_only": sorted(end_words - start_words)[:15],
    }
