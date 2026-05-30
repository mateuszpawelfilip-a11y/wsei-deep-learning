from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile


def resolve_device(preferred: str = "cuda") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def normalize_wav(wav: np.ndarray) -> np.ndarray:
    """Float32 mono w zakresie [-1, 1] do odtwarzania i zapisu WAV."""
    w = np.asarray(wav, dtype=np.float32).squeeze()
    if w.size == 0:
        raise ValueError("Pusta tablica audio")
    peak = np.max(np.abs(w))
    if peak > 1.0:
        w = w / peak
    return np.clip(w, -1.0, 1.0)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Wczytaj WAV (mono float32) bez torchaudio."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    return normalize_wav(data), int(sr)


def save_audio(wav: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = normalize_wav(wav)
    wavfile.write(str(path), int(sample_rate), (w * 32767).astype(np.int16))


def show_image_grid(images: list[Image.Image], titles: list[str], figsize=(14, 4)) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_run_meta(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def zip_outputs(source_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(source_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(source_dir.parent))
    return zip_path
