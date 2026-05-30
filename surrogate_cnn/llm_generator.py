import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .architecture import parse_llm_json, validate_architecture
from .config import (
    HF_MAX_NEW_TOKENS,
    HF_MODEL_NAME,
    HF_TEMPERATURE,
    MAX_CONV_LAYERS,
)
from .layers import normalize_layers

SYSTEM_PROMPT = """You generate CNN architectures for CIFAR-10 (32x32 RGB, 10 classes).
Return ONLY valid JSON with this schema:
{
  "layers": [
    {"type":"conv","filters":32,"kernel":3},
    {"type":"batchnorm"},
    {"type":"relu"},
    {"type":"maxpool"},
    {"type":"dropout","p":0.25},
    {"type":"globalaveragepooling"},
    {"type":"linear","units":128}
  ]
}
Allowed types: conv, batchnorm, relu, maxpool, dropout, linear, globalaveragepooling.
Rules: at most 6 conv layers, end with globalaveragepooling then linear, use kernel 3 or 5.
"""

USER_PROMPT = "Generate one diverse CNN architecture JSON."

_tokenizer = None
_model = None


def _llm_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_llm():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        _model = _model.to(_llm_device())
    _model.eval()
    return _tokenizer, _model


def llm_available() -> bool:
    try:
        import transformers
        return True
    except ImportError:
        return False


def _ollama_available() -> bool:
    return llm_available()


def _build_prompt(tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\nUser: {USER_PROMPT}\n\nAssistant:"


def _generate_text(
    temperature: float = HF_TEMPERATURE,
    max_new_tokens: int = HF_MAX_NEW_TOKENS,
) -> str:
    tokenizer, model = _load_llm()
    prompt = _build_prompt(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_architecture_llm(
    temperature: float = HF_TEMPERATURE,
    max_attempts: int = 3,
) -> dict | None:
    if not llm_available():
        return None
    for _ in range(max_attempts):
        content = _generate_text(temperature=temperature)
        arch = parse_llm_json(content)
        if arch is None:
            continue
        arch["layers"] = normalize_layers(arch.get("layers", []))
        ok, _ = validate_architecture(arch)
        if ok:
            return arch
    return None


def generate_architecture_random(max_attempts: int = 50) -> dict:
    conv_options = [16, 32, 48, 64, 96, 128]
    for _ in range(max_attempts):
        n_conv = random.randint(2, min(4, MAX_CONV_LAYERS))
        layers = []
        filters = random.choice(conv_options)
        for i in range(n_conv):
            layers.append({"type": "conv", "filters": filters, "kernel": random.choice([3, 5])})
            if random.random() < 0.7:
                layers.append({"type": "batchnorm"})
            layers.append({"type": "relu"})
            if i < n_conv - 1 and random.random() < 0.6:
                layers.append({"type": "maxpool"})
            filters = min(filters * 2, 256)
        if layers[-1]["type"] == "maxpool":
            layers.pop()
        layers.append({"type": "dropout", "p": round(random.uniform(0.1, 0.5), 2)})
        layers.append({"type": "globalaveragepooling"})
        layers.append({"type": "linear", "units": random.choice([64, 128, 256])})
        arch = {"layers": normalize_layers(layers)}
        ok, _ = validate_architecture(arch)
        if ok:
            return arch
    return {
        "layers": [
            {"type": "conv", "filters": 32, "kernel": 3},
            {"type": "relu"},
            {"type": "maxpool"},
            {"type": "globalaveragepooling"},
            {"type": "linear", "units": 128},
        ]
    }


def generate_architecture(use_llm: bool = True) -> tuple[dict, str]:
    if use_llm:
        arch = generate_architecture_llm()
        if arch is not None:
            return arch, "hf"
    return generate_architecture_random(), "random"
