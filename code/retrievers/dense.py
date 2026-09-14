"""Dense retriever with sentence-transformers all-MiniLM-L6-v2."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from .bm25 import Retrieved


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        self.encoder = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encoder.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    def retrieve(self, question: str, passages, k: int = 5) -> list[Retrieved]:
        if not passages:
            return []
        q = self.encode([question])[0]
        docs = self.encode([p.text for p in passages])
        scores = docs @ q
        order = sorted(range(len(passages)), key=lambda i: -float(scores[i]))
        out = []
        for i in order[:k]:
            p = passages[i]
            out.append(Retrieved(index=i, title=p.title, text=p.text, score=float(scores[i]), is_gold=p.is_gold))
        return out
