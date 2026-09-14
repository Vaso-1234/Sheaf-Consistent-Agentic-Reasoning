"""Factory that picks the right LLM class for a model name."""

from __future__ import annotations

from .base import LLM
from .flan_t5 import FlanT5LLM
from .qwen import QwenLLM


def build_llm(model_name: str, max_new_tokens: int = 32) -> LLM:
    lname = model_name.lower()
    if "flan-t5" in lname or "t5" in lname:
        return FlanT5LLM(model_name, max_new_tokens=max_new_tokens)
    if "qwen" in lname:
        return QwenLLM(model_name, max_new_tokens=max_new_tokens)
    raise ValueError(f"no wrapper for model {model_name}")


LLM_LABELS = {
    "google/flan-t5-base": "FLAN-T5-Base (250M)",
    "google/flan-t5-large": "FLAN-T5-Large (780M)",
    "google/flan-t5-xl": "FLAN-T5-XL (3B)",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
}
