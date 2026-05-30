from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "lab04"
ZIP_PATH = ROOT / "outputs" / "lab04_wsei.zip"

FAST_MODE = False

DEVICE = "cuda" 

# --- Obrazy (diffusers) ---
IMAGE_MODELS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd_turbo": "stabilityai/sd-turbo",
}

IMAGE_STEPS = {"sd15": 30, "sd_turbo": 4}
GUIDANCE = {"sd15": 7.5, "sd_turbo": 0.0}

# --- Muzyka (transformers MusicGen; NIE audiocraft na Colab) ---
MUSIC_MODELS = {
    "musicgen_small": "facebook/musicgen-small",
    "musicgen_medium": "facebook/musicgen-medium",
}

MUSIC_DURATION_S = 8 if FAST_MODE else 15
MUSIC_MAX_NEW_TOKENS = 256 if FAST_MODE else 512

# --- Caption / audio → tekst ---
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
AUDIO_CLASSIFIER = "MIT/ast-finetuned-audioset-10-10-0.4593"

PIPELINE_RUNS = 2 if FAST_MODE else 3

WSEI_IMAGE_PROMPTS = [
    "Modern WSEI university campus in Krakow Poland, glass buildings, students walking, sunny day, photorealistic",
    "IDEIS DSW WSEI Krakow academic building interior, library, laptops, warm lighting, professional photo",
    "WSEI Krakow graduation day, crowd of students, flags, celebratory atmosphere, cinematic",
]

WSEI_MUSIC_PROMPTS = [
    "upbeat orchestral university anthem, inspiring, brass and strings",
    "ambient electronic study music, calm campus atmosphere, soft pads",
    "energetic rock marching band, student parade, festive university event",
]
