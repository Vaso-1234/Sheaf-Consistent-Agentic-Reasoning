"""Shared LLM interface used by every solver."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Generation:
    text: str
    tokens: int


_norm_re = re.compile(r"[^a-z0-9\s]")
_ws = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    return _ws.sub(" ", _norm_re.sub(" ", (s or "").lower())).strip()


def extract_short_answer(text: str) -> str:
    """Pull out the first meaningful short answer from a generation."""
    if text is None:
        return ""
    text = text.strip()
    # Explicit "answer:" pattern
    for pat in [r"(?:short answer|final answer|answer)\s*[:\-]\s*(.+?)(?:\n|$)",
                r"^([^\n]+)"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip().rstrip(".").rstrip(",").strip('"').strip("'")
            if cand:
                return cand[:120]
    return text[:120]


class LLM:
    """Abstract interface. Subclasses implement _generate_one."""

    name: str = "abstract"

    def generate_n(self, prompt: str, n: int = 1, seed: int = 42,
                   temperature: float = 0.9, top_p: float = 0.95,
                   max_new_tokens: Optional[int] = None) -> list[Generation]:
        return [self._generate_one(prompt, seed + i, temperature=temperature, top_p=top_p,
                                   max_new_tokens=max_new_tokens)
                for i in range(n)]

    def generate_greedy(self, prompt: str, max_new_tokens: Optional[int] = None) -> Generation:
        return self._generate_one(prompt, seed=0, temperature=0.0, top_p=1.0,
                                  max_new_tokens=max_new_tokens, greedy=True)

    def _generate_one(self, prompt: str, seed: int, temperature: float, top_p: float,
                      max_new_tokens: Optional[int], greedy: bool = False) -> Generation:
        raise NotImplementedError
