"""Semantic Generation Module backends: BLIP-2, GIT, and LLaVA captioners.

The paper evaluates three image-to-text models (Table 5); the original code
shipped only the BLIP-2 scripts. All three are implemented here behind one
interface, selected via `captioning.backend` in the config.

Notes:
  - Quantization (`8bit`/`4bit`) requires the optional `bitsandbytes` package
    and roughly quarters/halves GPU memory, enabling large checkpoints such
    as blip2-flan-t5-xxl on consumer GPUs.
  - GIT is a pure captioning model and does not condition on a text prompt;
    the configured prompt is ignored for that backend (see REFACTORING_NOTES).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _model_kwargs(quantization: str, dtype: str, device: str) -> dict:
    kwargs: dict = {}
    if quantization in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "8bit",
            load_in_4bit=quantization == "4bit",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype) if device != "cpu" else torch.float32
        kwargs["device_map"] = "auto" if device != "cpu" else None
    return kwargs


class Captioner:
    """Base class: load once, caption image batches deterministically."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        quantization: str = "none",
        dtype: str = "float16",
        max_new_tokens: int = 100,
        prompt: str = "",
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._load(_model_kwargs(quantization, dtype, device))

    def _load(self, model_kwargs: dict) -> None:
        raise NotImplementedError

    def caption_batch(self, images: list[Image.Image]) -> list[str]:
        raise NotImplementedError

    def _generate(self, inputs) -> list[str]:
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]


class Blip2Captioner(Captioner):
    def _load(self, model_kwargs: dict) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
        if model_kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()

    def caption_batch(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(
            images=images, text=[self.prompt] * len(images), return_tensors="pt", padding=True
        ).to(self.model.device)
        return self._generate(inputs)


class GitCaptioner(Captioner):
    def _load(self, model_kwargs: dict) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if model_kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()

    def caption_batch(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        return self._generate(inputs)


class LlavaCaptioner(Captioner):
    def _load(self, model_kwargs: dict) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
        if model_kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()

    def caption_batch(self, images: list[Image.Image]) -> list[str]:
        chat = f"USER: <image>\n{self.prompt} ASSISTANT:"
        inputs = self.processor(
            images=images, text=[chat] * len(images), return_tensors="pt", padding=True
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [text.rsplit("ASSISTANT:", 1)[-1].strip() for text in texts]


class BlipCaptioner(Captioner):
    """Small BLIP-1 captioner (~470M params); used by the smoke test."""

    def _load(self, model_kwargs: dict) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
        if model_kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()

    def caption_batch(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        return self._generate(inputs)


_BACKENDS = {
    "blip2": Blip2Captioner,
    "git": GitCaptioner,
    "llava": LlavaCaptioner,
    "blip": BlipCaptioner,
}


def build_captioner(cfg: dict, device: str) -> Captioner:
    backend = cfg.get("backend", "blip2")
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown captioning backend {backend!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[backend](
        model_name=cfg["model"],
        device=device,
        quantization=cfg.get("quantization", "none"),
        dtype=cfg.get("dtype", "float16"),
        max_new_tokens=cfg.get("max_new_tokens", 100),
        prompt=cfg.get("prompt", ""),
    )


def load_image(path: str | Path) -> Image.Image | None:
    """Open + verify an image, returning None (not raising) on corrupt files."""
    try:
        with Image.open(path) as probe:
            probe.verify()
        return Image.open(path).convert("RGB")
    except Exception:
        return None
