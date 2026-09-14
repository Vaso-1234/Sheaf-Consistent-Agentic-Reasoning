"""FLAN-T5 (seq2seq) wrapper. Cached checkpoints: base, large, xl."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import LLM, Generation


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FlanT5LLM(LLM):
    def __init__(self, model_name: str = "google/flan-t5-large",
                 max_new_tokens: int = 32,
                 use_fp16: Optional[bool] = None):
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = _device()
        if use_fp16 is None:
            use_fp16 = any(t in model_name for t in ("large", "xl", "xxl")) and self.device in ("cuda", "mps")
        kwargs: dict = {}
        if use_fp16:
            kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _generate_one(self, prompt: str, seed: int, temperature: float, top_p: float,
                      max_new_tokens: Optional[int], greedy: bool = False) -> Generation:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].numel())
        gen_kwargs = {"max_new_tokens": max_new_tokens or self.max_new_tokens}
        if greedy or temperature <= 0.0:
            gen_kwargs.update(do_sample=False, num_return_sequences=1)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=1)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        total = prompt_tokens + int(out.shape[-1])
        return Generation(text=text, tokens=total)
