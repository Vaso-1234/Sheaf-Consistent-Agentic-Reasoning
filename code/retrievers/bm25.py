"""BM25 retriever over a per-example passage pool."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi


_tok_re = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _tok_re.findall(text or "")]


@dataclass
class Retrieved:
    index: int
    title: str
    text: str
    score: float
    is_gold: bool


class BM25Retriever:
    """Fresh BM25 index for each query's passage pool. Cheap because pools are 10-20 docs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def retrieve(self, question: str, passages: Sequence, k: int = 5) -> list[Retrieved]:
        docs = [p.text for p in passages]
        tokens = [tokenize(d) for d in docs]
        if not any(tokens):
            return []
        bm25 = BM25Okapi(tokens, k1=self.k1, b=self.b)
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(passages)), key=lambda i: -float(scores[i]))
        out = []
        for i in order[:k]:
            p = passages[i]
            out.append(Retrieved(index=i, title=p.title, text=p.text, score=float(scores[i]), is_gold=p.is_gold))
        return out
