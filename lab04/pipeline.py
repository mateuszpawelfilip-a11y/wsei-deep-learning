from __future__ import annotations

from pathlib import Path

from .caption import audio_to_caption, compare_prompts, describe_image
from .config import OUTPUT_DIR
from .images import generate_image
from .music import generate_music
from .utils import save_audio, save_image, save_run_meta


def run_wsei_pipeline(
    start_prompt: str,
    music_prompt: str,
    run_id: int,
    device: str,
    image_model: str = "sd15",
    music_model: str = "musicgen_small",
    seed: int = 42,
) -> dict:
    """
    1. prompt → obraz
    2. BLIP opis obrazu
    3. nowy obraz z opisu
    4. muzyka (wspólna dla obu)
    5. obraz z muzyki (AST → caption → SD)
    6. BLIP opis końcowego obrazu
    7. porównanie start vs koniec
    """
    run_dir = OUTPUT_DIR / f"run_{run_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1–2
    img1 = generate_image(start_prompt, model_key=image_model, device=device, seed=seed)
    save_image(img1, run_dir / "01_image_from_prompt.png")
    caption1 = describe_image(img1, device)

    # 3
    img2 = generate_image(caption1, model_key=image_model, device=device, seed=seed + 1)
    save_image(img2, run_dir / "03_image_from_caption.png")

    # 4
    wav, sr = generate_music(music_prompt, model_key=music_model, device=device)
    music_path = run_dir / "04_music.wav"
    save_audio(wav, sr, music_path)

    # 5 (music → text → image)
    music_caption = audio_to_caption(music_path, device)
    img_from_music = generate_image(
        music_caption, model_key=image_model, device=device, seed=seed + 2
    )
    save_image(img_from_music, run_dir / "06_image_from_music.png")

    # 6–7
    final_caption = describe_image(img_from_music, device)
    comparison = compare_prompts(start_prompt, final_caption)

    meta = {
        "run_id": run_id,
        "start_prompt": start_prompt,
        "music_prompt": music_prompt,
        "image_model": image_model,
        "music_model": music_model,
        "caption_after_img1": caption1,
        "music_to_image_caption": music_caption,
        "final_caption": final_caption,
        "comparison": comparison,
        "files": [
            "01_image_from_prompt.png",
            "03_image_from_caption.png",
            "04_music.wav",
            "06_image_from_music.png",
        ],
    }
    save_run_meta(run_dir, meta)
    return meta
