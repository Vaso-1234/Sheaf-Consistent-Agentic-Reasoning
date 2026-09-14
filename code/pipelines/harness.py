"""Batch runner that shares LLM samples across all methods for efficiency.

For each Example we do:
    dense = DenseRetriever(top-5)
    bm25  = BM25Retriever(top-5)
    triples = build_subgraph(top-10 dense)
    N samples once via LLM.generate_n
    Then reuse samples for self-consistency, process_reward, scar variants.
    Two extra greedy calls for single_shot and graphrag_approx.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bootstrap_ci, dump_json, em_score, f1_score, loose_em, normalize_text
from data_loaders.registry import DATASET_LABELS, load_examples
from kg.subgraph_extractor import build_subgraph
from llm.registry import build_llm
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from verifier.sheaf import SheafVerifier, VerifierConfig
from baselines.methods import (
    graphrag_approx,
    process_reward,
    scar,
    self_consistency,
    single_shot,
)
from baselines.solver import RetrievalBundle, build_context_from_passages, build_context_from_triples, qa_prompt


METHOD_LIST_DEFAULT = [
    "single_shot",
    "self_consistency",
    "graphrag_approx",
    "process_reward",
    "scar_identity",
    "scar_llm",
]


def _numeric_hit(pred: str, gold: str) -> float:
    """PolicyBench answers look like '30 days'. Accept exact numeric match."""
    import re

    p = normalize_text(pred)
    g = normalize_text(gold)
    p_nums = re.findall(r"\d+", p)
    g_nums = re.findall(r"\d+", g)
    if g_nums and p_nums and p_nums[0] == g_nums[0]:
        return 1.0
    return 0.0


def score_answer(pred: str, gold: str, aliases: Sequence[str], dataset: str) -> dict:
    em = max(em_score(pred, gold), *(em_score(pred, a) for a in aliases or [gold]))
    f1 = max(f1_score(pred, gold), *(f1_score(pred, a) for a in aliases or [gold]))
    lem = max(loose_em(pred, gold), *(loose_em(pred, a) for a in aliases or [gold]))
    hit = lem
    if dataset == "policybench":
        hit = max(hit, _numeric_hit(pred, gold))
    return {"em": em, "f1": f1, "loose_em": lem, "hit": hit}


def grounded_hit(pred_titles: Sequence[str], gold_titles: Sequence[str], answer_correct: float) -> float:
    if answer_correct < 1.0:
        return 0.0
    pred = set(pred_titles or [])
    gold = set(gold_titles or [])
    if not gold:
        return answer_correct
    return 1.0 if bool(pred & gold) else 0.0


def _retrieval_bundle(question: str, example, bm25, dense) -> RetrievalBundle:
    dense_top = dense.retrieve(question, example.passages, k=5)
    bm25_top = bm25.retrieve(question, example.passages, k=5)
    # For triples, use union of dense top-8 + bm25 top-4 to give diverse sources
    seen_i = set()
    union_pool = []
    for r in list(dense_top) + list(bm25_top):
        if r.index in seen_i:
            continue
        seen_i.add(r.index)
        union_pool.append(type("P", (), {"title": r.title, "text": r.text})())
    triples = build_subgraph(union_pool[:10], top_k=10)
    return RetrievalBundle(dense_passages=dense_top, bm25_passages=bm25_top, subgraph_triples=triples)


def run_one_example(
    example, llm, bm25, dense, verifiers: dict, methods: Sequence[str],
    n_samples: int = 5, seed: int = 42, max_new_tokens: int = 32,
) -> dict:
    bundle = _retrieval_bundle(example.question, example, bm25, dense)

    # Shared sampled candidates for methods that need them.
    ctx_str = build_context_from_passages(bundle.dense_passages, max_chars=1600)
    prompt = qa_prompt(example.question, ctx_str)
    t0 = time.time()
    gens = llm.generate_n(prompt, n=n_samples, seed=seed, max_new_tokens=max_new_tokens)
    sampling_ms = (time.time() - t0) * 1000.0
    from llm.base import extract_short_answer
    shared_cands = [extract_short_answer(g.text).strip() for g in gens]
    shared_tokens = sum(g.tokens for g in gens)

    preds: dict[str, dict] = {}
    # We want SCAR to see the greedy answer as an anchor; run single_shot first
    # if either it or a scar_* method is requested, and remember its answer.
    anchor_answer: Optional[str] = None
    ordered_methods = list(methods)
    if any(m.startswith("scar_") for m in methods) and "single_shot" in methods:
        ordered_methods = ["single_shot"] + [m for m in methods if m != "single_shot"]

    for m in ordered_methods:
        if m == "single_shot":
            p = single_shot(example.question, bundle, llm, max_new_tokens=max_new_tokens)
            anchor_answer = p.answer
        elif m == "self_consistency":
            # Reuse shared_cands
            top, _ = Counter(shared_cands).most_common(1)[0]
            from baselines.solver import Prediction
            p = Prediction(
                method="self_consistency",
                answer=top,
                candidates=shared_cands,
                used_evidence_titles=[x.title for x in bundle.dense_passages],
                tokens=shared_tokens,
                latency_ms=sampling_ms,
            )
        elif m == "graphrag_approx":
            p = graphrag_approx(example.question, bundle, llm, max_new_tokens=max_new_tokens)
        elif m == "process_reward":
            # Reuse shared_cands but still self-score via the LLM
            p = _process_reward_reuse(example.question, bundle, llm, shared_cands, shared_tokens, sampling_ms)
        elif m.startswith("scar_"):
            variant = m.split("_", 1)[1]
            ver = verifiers[variant]
            p = _scar_reuse(example.question, bundle, ver, shared_cands, shared_tokens, sampling_ms,
                            method_tag=m, anchor=anchor_answer)
        else:
            raise ValueError(f"unknown method: {m}")

        scores = score_answer(p.answer, example.answer, example.answer_aliases, example.dataset)
        gh = grounded_hit(p.used_evidence_titles, example.supporting_titles, scores["hit"])
        preds[m] = {
            "answer": p.answer,
            "candidates": p.candidates,
            "used_titles": p.used_evidence_titles,
            "used_sheaf": p.used_sheaf,
            "tokens": p.tokens,
            "latency_ms": p.latency_ms,
            "em": scores["em"],
            "f1": scores["f1"],
            "loose_em": scores["loose_em"],
            "hit": scores["hit"],
            "grounded_hit": gh,
            "diagnostics": p.diagnostics,
        }

    return {
        "qid": example.qid,
        "dataset": example.dataset,
        "qtype": example.qtype,
        "question": example.question,
        "gold": example.answer,
        "supporting_titles": example.supporting_titles,
        "shared_candidates": shared_cands,
        "sampling_ms": sampling_ms,
        "preds": preds,
    }


def _process_reward_reuse(question, bundle, llm, cands, tokens_used, sampling_ms):
    """Score existing candidates for evidence support via the LLM (avoid re-sampling)."""
    from baselines.solver import Prediction, build_context_from_passages
    ctx = build_context_from_passages(bundle.dense_passages, max_chars=1600)
    t0 = time.time()
    scores = []
    tokens = tokens_used
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
    best_score = max(scores) if scores else 0.0
    best_cands = [c for c, s in zip(cands, scores) if s == best_score] or cands
    winner = Counter(best_cands).most_common(1)[0][0]
    return Prediction(
        method="process_reward",
        answer=winner,
        candidates=cands,
        used_evidence_titles=[x.title for x in bundle.dense_passages],
        tokens=tokens,
        latency_ms=sampling_ms + (time.time() - t0) * 1000.0,
        diagnostics={"scores": scores},
    )


def _scar_reuse(question, bundle, verifier, cands, tokens_used, sampling_ms, method_tag: str,
                anchor: Optional[str] = None, extra_candidates: Optional[Sequence[str]] = None):
    from baselines.solver import Prediction
    passages_text = [f"[{p.title}] {p.text}" for p in bundle.dense_passages]
    bm25_text = [f"[{p.title}] {p.text}" for p in bundle.bm25_passages]
    triples_text = [t.as_text() for t in bundle.subgraph_triples]
    src = {
        "passages": verifier.build_source("passages", passages_text, "passage"),
        "triples": verifier.build_source("triples", triples_text, "triple"),
        "tool": verifier.build_source("tool", bm25_text, "tool"),
    }
    t0 = time.time()
    res = verifier.verify(question, cands, src, anchor=anchor,
                          extra_candidates=extra_candidates)
    veri_ms = (time.time() - t0) * 1000.0
    used_titles = [p.title for p in bundle.dense_passages] + \
                  [p.title for p in bundle.bm25_passages if p.title not in {q.title for q in bundle.dense_passages}]
    return Prediction(
        method=method_tag,
        answer=res.chosen_answer,
        candidates=cands,
        used_evidence_titles=used_titles,
        used_sheaf=res.used_sheaf,
        tokens=tokens_used,
        latency_ms=sampling_ms + veri_ms,
        diagnostics={
            "energies": res.energies,
            "final_scores": res.final_scores,
            "vote_shares": res.vote_shares,
            "alpha": res.alpha,
            "vote_pick": res.vote_pick,
            "verifier_ms": veri_ms,
        },
    )


def build_verifiers(encoder, gate_enabled: bool = True) -> dict:
    return {
        "identity": SheafVerifier(encoder, VerifierConfig(restriction="identity", gate_enabled=gate_enabled)),
        "llm": SheafVerifier(encoder, VerifierConfig(restriction="llm", gate_enabled=gate_enabled)),
        "linear": SheafVerifier(encoder, VerifierConfig(restriction="linear", gate_enabled=gate_enabled)),
    }


def aggregate(records: list[dict], methods: Sequence[str]) -> dict:
    import numpy as np

    summary = {"n": len(records), "methods": {}}
    for m in methods:
        em = [r["preds"][m]["em"] for r in records]
        f1 = [r["preds"][m]["f1"] for r in records]
        lem = [r["preds"][m]["loose_em"] for r in records]
        hit = [r["preds"][m]["hit"] for r in records]
        gh = [r["preds"][m]["grounded_hit"] for r in records]
        lat = [r["preds"][m]["latency_ms"] for r in records]
        toks = [r["preds"][m]["tokens"] for r in records]
        summary["methods"][m] = {
            "em_mean": float(np.mean(em)),
            "em_ci95": list(bootstrap_ci(em)),
            "f1_mean": float(np.mean(f1)),
            "loose_em_mean": float(np.mean(lem)),
            "hit_mean": float(np.mean(hit)),
            "hit_ci95": list(bootstrap_ci(hit)),
            "grounded_mean": float(np.mean(gh)),
            "grounded_ci95": list(bootstrap_ci(gh)),
            "latency_ms_mean": float(np.mean(lat)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "tokens_mean": float(np.mean(toks)),
        }
    return summary
