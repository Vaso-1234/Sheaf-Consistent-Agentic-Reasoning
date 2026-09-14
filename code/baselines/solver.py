"""Shared solver protocol: input is (Example, Retrieval bundle, LLM), output is
a Prediction with answer, evidence, and per-method diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalBundle:
    dense_passages: list          # list[Retrieved] top-5 dense
    bm25_passages: list           # list[Retrieved] top-5 bm25
    subgraph_triples: list        # list[Triple]
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Prediction:
    method: str
    answer: str
    candidates: list = field(default_factory=list)
    used_evidence_titles: list = field(default_factory=list)
    used_sheaf: bool = False
    tokens: int = 0
    latency_ms: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def build_context_from_passages(passages, max_chars: int = 1600) -> str:
    parts = []
    total = 0
    for p in passages:
        block = f"[{p.title}] {p.text}"
        if total + len(block) > max_chars:
            block = block[: max(0, max_chars - total)]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n".join(parts)


def build_context_from_triples(triples, max_lines: int = 20) -> str:
    lines = []
    for t in triples[:max_lines]:
        lines.append(t.as_text() if hasattr(t, "as_text") else str(t))
    return "\n".join(lines) if lines else "(no triples)"


def qa_prompt(question: str, context: str) -> str:
    return (
        "Read the context and answer the question with a short phrase only.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Short answer:"
    )


def graphrag_prompt(question: str, passages_ctx: str, triples_ctx: str) -> str:
    return (
        "Use the passages and knowledge-graph triples to answer.\n"
        f"Passages:\n{passages_ctx}\n\n"
        f"Triples:\n{triples_ctx}\n\n"
        f"Question: {question}\n"
        "Short answer:"
    )
