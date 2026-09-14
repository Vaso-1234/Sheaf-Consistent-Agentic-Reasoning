"""All baselines + SCAR entry point behind one function each."""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from llm.base import LLM, extract_short_answer

from .solver import Prediction, RetrievalBundle, build_context_from_passages, build_context_from_triples, graphrag_prompt, qa_prompt


def _clean(a: str) -> str:
    return extract_short_answer(a).strip()


def single_shot(question: str, bundle: RetrievalBundle, llm: LLM,
                max_new_tokens: int = 32) -> Prediction:
    t0 = time.time()
    ctx = build_context_from_passages(bundle.dense_passages, max_chars=1600)
    prompt = qa_prompt(question, ctx)
    g = llm.generate_greedy(prompt, max_new_tokens=max_new_tokens)
    ans = _clean(g.text)
    return Prediction(
        method="single_shot",
        answer=ans,
        candidates=[ans],
        used_evidence_titles=[p.title for p in bundle.dense_passages],
        tokens=g.tokens,
        latency_ms=(time.time() - t0) * 1000.0,
    )


def self_consistency(question: str, bundle: RetrievalBundle, llm: LLM,
                     n_samples: int = 5, seed: int = 42,
                     max_new_tokens: int = 32) -> Prediction:
    t0 = time.time()
    ctx = build_context_from_passages(bundle.dense_passages, max_chars=1600)
    prompt = qa_prompt(question, ctx)
    gens = llm.generate_n(prompt, n=n_samples, seed=seed, max_new_tokens=max_new_tokens)
    cands = [_clean(g.text) for g in gens]
    top, _ = Counter(cands).most_common(1)[0]
    return Prediction(
        method="self_consistency",
        answer=top,
        candidates=cands,
        used_evidence_titles=[p.title for p in bundle.dense_passages],
        tokens=sum(g.tokens for g in gens),
        latency_ms=(time.time() - t0) * 1000.0,
    )


def graphrag_approx(question: str, bundle: RetrievalBundle, llm: LLM,
                    max_new_tokens: int = 32) -> Prediction:
    """Passages + extracted triples in the prompt, single greedy decode."""
    t0 = time.time()
    ctx_p = build_context_from_passages(bundle.dense_passages[:4], max_chars=1200)
    ctx_t = build_context_from_triples(bundle.subgraph_triples, max_lines=16)
    prompt = graphrag_prompt(question, ctx_p, ctx_t)
    g = llm.generate_greedy(prompt, max_new_tokens=max_new_tokens)
    ans = _clean(g.text)
    return Prediction(
        method="graphrag_approx",
        answer=ans,
        candidates=[ans],
        used_evidence_titles=[p.title for p in bundle.dense_passages[:4]],
        tokens=g.tokens,
        latency_ms=(time.time() - t0) * 1000.0,
    )


def process_reward(question: str, bundle: RetrievalBundle, llm: LLM,
                   n_samples: int = 5, seed: int = 42,
                   max_new_tokens: int = 32) -> Prediction:
    """N samples then self-score each for evidence support, take argmax."""
    t0 = time.time()
    ctx = build_context_from_passages(bundle.dense_passages, max_chars=1600)
    prompt = qa_prompt(question, ctx)
    gens = llm.generate_n(prompt, n=n_samples, seed=seed, max_new_tokens=max_new_tokens)
    cands = [_clean(g.text) for g in gens]
    scores = []
    tokens = sum(g.tokens for g in gens)
    for c in cands:
        judge = (
            "You are a strict grader. Given the passages, does the answer "
            "correctly answer the question using the passages? Reply yes or no.\n"
            f"Passages:\n{ctx}\n\n"
            f"Question: {question}\n"
            f"Answer: {c}\n"
            "Grade:"
        )
        jg = llm.generate_greedy(judge, max_new_tokens=6)
        tokens += jg.tokens
        text = jg.text.lower()
        if "yes" in text and "no" not in text[:6]:
            scores.append(1.0)
        elif "no" in text:
            scores.append(0.0)
        else:
            scores.append(0.5)
    # tie-break by vote share
    counts = Counter(cands)
    best_score = max(scores)
    best_cands = [c for c, s in zip(cands, scores) if s == best_score]
    winner = Counter(best_cands).most_common(1)[0][0]
    return Prediction(
        method="process_reward",
        answer=winner,
        candidates=cands,
        used_evidence_titles=[p.title for p in bundle.dense_passages],
        tokens=tokens,
        latency_ms=(time.time() - t0) * 1000.0,
        diagnostics={"scores": scores},
    )


def scar(question: str, bundle: RetrievalBundle, llm: LLM,
         sheaf_verifier,
         n_samples: int = 5, seed: int = 42,
         max_new_tokens: int = 32,
         method_tag: str = "scar") -> Prediction:
    """SCAR: N-sample decode then sheaf-consistency scoring across 3 sources.

    method_tag lets us log ablations (e.g. scar_identity, scar_llm, scar_no_gate)
    while sharing this function.
    """
    t0 = time.time()
    ctx = build_context_from_passages(bundle.dense_passages, max_chars=1600)
    prompt = qa_prompt(question, ctx)
    gens = llm.generate_n(prompt, n=n_samples, seed=seed, max_new_tokens=max_new_tokens)
    cands = [_clean(g.text) for g in gens]
    tokens = sum(g.tokens for g in gens)

    # Build the three evidence sources.
    passages_text = [f"[{p.title}] {p.text}" for p in bundle.dense_passages]
    bm25_text = [f"[{p.title}] {p.text}" for p in bundle.bm25_passages]
    triples_text = [t.as_text() for t in bundle.subgraph_triples]

    src = {
        "passages": sheaf_verifier.build_source("passages", passages_text, "passage"),
        "triples": sheaf_verifier.build_source("triples", triples_text, "triple"),
        "tool": sheaf_verifier.build_source("tool", bm25_text, "tool"),
    }

    res = sheaf_verifier.verify(question, cands, src)
    used_titles = [p.title for p in bundle.dense_passages] + \
                  [p.title for p in bundle.bm25_passages if p.title not in {q.title for q in bundle.dense_passages}]
    return Prediction(
        method=method_tag,
        answer=res.chosen_answer,
        candidates=cands,
        used_evidence_titles=used_titles,
        used_sheaf=res.used_sheaf,
        tokens=tokens,
        latency_ms=(time.time() - t0) * 1000.0,
        diagnostics={
            "energies": res.energies,
            "final_scores": res.final_scores,
            "vote_shares": res.vote_shares,
            "alpha": res.alpha,
            "vote_pick": res.vote_pick,
            "verifier_ms": res.latency_ms,
        },
    )
