"""LLM-based Semantic Fusion Module backends.

Two model families, selected via `fusion.backend`:
  - "seq2seq": encoder-decoder models (Flan-T5) fed the paper's plain prompt.
  - "causal":  chat models (Mistral-7B-Instruct, Llama-3.1-8B-Instruct) fed a
               system+user conversation via the tokenizer's chat template.

Decoding is configurable. The original Mistral/Llama scripts sampled with
temperature 0.6 (non-reproducible); `deterministic: true` switches to greedy /
beam decoding so identical inputs always yield identical summaries.
"""

from __future__ import annotations

import torch

PAPER_PROMPT = (
    "Your task is to integrate the following list of visual descriptions for the "
    "entity '{entity_name}' into a rich, detailed, and coherent summary paragraph. "
    "Capture as many key details as possible, such as objects, colors, actions, "
    "and settings. Your final output must be a single paragraph, not a list."
)

CHAT_SYSTEM_PROMPT = (
    "You are an expert in summarizing and synthesizing information. Your task is "
    "to integrate a list of visual descriptions about an entity into a single, "
    "rich, and detailed paragraph. Capture as many key details as possible and "
    "ensure the final output is a coherent paragraph, not a list."
)


def _quantization_kwargs(quantization: str, dtype: str) -> dict:
    if quantization in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=quantization == "8bit",
                load_in_4bit=quantization == "4bit",
            ),
            "device_map": "auto",
        }
    return {"dtype": getattr(torch, dtype), "device_map": "auto"}


class Fuser:
    def __init__(self, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        self.max_new_tokens = cfg.get("max_new_tokens", 384)
        self.deterministic = cfg.get("deterministic", True)
        self._load()

    def _load(self) -> None:
        raise NotImplementedError

    def fuse(self, entity_name: str, descriptions: list[str]) -> str:
        raise NotImplementedError

    def _decoding_kwargs(self) -> dict:
        if self.deterministic:
            return {"do_sample": False, "num_beams": self.cfg.get("num_beams", 1)}
        return {
            "do_sample": True,
            "temperature": self.cfg.get("temperature", 0.6),
            "top_p": self.cfg.get("top_p", 0.9),
        }


class Seq2SeqFuser(Fuser):
    def _load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = self.cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = _quantization_kwargs(self.cfg.get("quantization", "none"), self.cfg.get("dtype", "float32"))
        if self.device == "cpu":
            kwargs = {"dtype": torch.float32}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        if kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()
        self.max_input_tokens = self.cfg.get("max_input_tokens", 1024)

    def fuse(self, entity_name: str, descriptions: list[str]) -> str:
        prompt = (
            PAPER_PROMPT.format(entity_name=entity_name)
            + "\n\nList of descriptions to summarize:\n"
            + "\n".join(descriptions)
            + "\n\nDetailed Summary Paragraph:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                early_stopping=self.cfg.get("num_beams", 1) > 1,
                **self._decoding_kwargs(),
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class CausalFuser(Fuser):
    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = _quantization_kwargs(self.cfg.get("quantization", "none"), self.cfg.get("dtype", "bfloat16"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def fuse(self, entity_name: str, descriptions: list[str]) -> str:
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Please synthesize the following descriptions for the entity "
                    f"'{entity_name}':\n\n" + "\n".join(descriptions)
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        eos_ids = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0 and eot != self.tokenizer.unk_token_id:
            eos_ids.append(eot)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **self._decoding_kwargs(),
            )
        response = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(response, skip_special_tokens=True).strip()


def build_fuser(cfg: dict, device: str) -> Fuser:
    backend = cfg.get("backend", "seq2seq")
    if backend == "seq2seq":
        return Seq2SeqFuser(cfg, device)
    if backend == "causal":
        return CausalFuser(cfg, device)
    raise ValueError(f"Unknown fusion backend {backend!r}; choose seq2seq or causal")
