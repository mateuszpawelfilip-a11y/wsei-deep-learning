from __future__ import annotations

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from .config import GUIDANCE, IMAGE_MODELS, IMAGE_STEPS


_pipes: dict[str, AutoPipelineForText2Image] = {}


def get_image_pipe(model_key: str, device: str) -> AutoPipelineForText2Image:
    if model_key not in IMAGE_MODELS:
        raise KeyError(f"Unknown model_key: {model_key}")
    if model_key not in _pipes:
        model_id = IMAGE_MODELS[model_key]
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        _pipes[model_key] = pipe
    return _pipes[model_key]


def generate_image(
    prompt: str,
    model_key: str = "sd15",
    device: str = "cuda",
    seed: int = 42,
    negative_prompt: str = "blurry, low quality, distorted text, watermark",
) -> Image.Image:
    pipe = get_image_pipe(model_key, device)
    steps = IMAGE_STEPS[model_key]
    guidance = GUIDANCE[model_key]
    generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if model_key == "sd_turbo":
        kwargs["guidance_scale"] = guidance
    else:
        kwargs["guidance_scale"] = guidance
        kwargs["negative_prompt"] = negative_prompt

    result = pipe(**kwargs)
    return result.images[0]


def unload_image_pipes() -> None:
    global _pipes
    for p in _pipes.values():
        del p
    _pipes = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
