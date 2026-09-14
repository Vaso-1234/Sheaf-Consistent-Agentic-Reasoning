"""Qwen 2.5 Instruct wrapper (causal LM with chat template)."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLM, Generation


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. Answer the user's question "
    "using ONLY the provided evidence. Reply with a short answer only, without "
    "explanation."
)


class QwenLLM(LLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 max_new_tokens: int = 32,
                 use_fp16: bool = False,
                 device_override: Optional[str] = "cpu"):
        """Qwen 2.5 wrapper. Defaults to CPU/FP32 because torch 2.5 MPS has a
        Qwen matmul dimensionality bug (mps_matmul: incompatible dimensions).
        """
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device_override or _device()
        kwargs: dict = {}
        if use_fp16 and self.device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _format(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_one(self, prompt: str, seed: int, temperature: float, top_p: float,
                      max_new_tokens: Optional[int], greedy: bool = False) -> Generation:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        chat = self._format(prompt)
        inputs = self.tokenizer(chat, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs = {"max_new_tokens": max_new_tokens or self.max_new_tokens,
                      "pad_token_id": self.tokenizer.eos_token_id}
        if greedy or temperature <= 0.0:
            gen_kwargs.update(do_sample=False, num_return_sequences=1)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=1)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        total = prompt_len + int(gen_ids.shape[-1])
        return Generation(text=text, tokens=total)
