#!/usr/bin/env python3
"""Ablations on HotpotQA subset: restriction map, gate, N samples.

Reuses stored jsonl records from a completed main run to avoid re-sampling.
If no cached run exists, samples from scratch.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sentence_transformers import SentenceTransformer

from common import OUTPUTS_DIR, bootstrap_ci, dump_json, set_all_seeds
from data_loaders.registry import load_examples
from kg.subgraph_extractor import build_subgraph
from llm.registry import build_llm
from pipelines.harness import _retrieval_bundle, score_answer, grounded_hit
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from verifier.sheaf import SheafVerifier, VerifierConfig


def slug(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _sheaf_pick(question, bundle, cands, verifier):
    from llm.base import extract_short_answer
    passages_text = [f"[{p.title}] {p.text}" for p in bundle.dense_passages]
    bm25_text = [f"[{p.title}] {p.text}" for p in bundle.bm25_passages]
    triples_text = [t.as_text() for t in bundle.subgraph_triples]
    src = {
        "passages": verifier.build_source("passages", passages_text, "passage"),
        "triples": verifier.build_source("triples", triples_text, "triple"),
        "tool": verifier.build_source("tool", bm25_text, "tool"),
    }
    res = verifier.verify(question, cands, src)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-large")
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    print(f"[load] dataset={args.dataset} n={args.n}", flush=True)
    examples = load_examples(args.dataset, n=args.n, seed=args.seed)
    print(f"[load] LLM={args.model}", flush=True)
    llm = build_llm(args.model, max_new_tokens=48)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    # Ablation configurations.
    configs = []
    for r in ["identity", "linear", "llm"]:
        configs.append(("restriction", r, VerifierConfig(restriction=r, gate_enabled=True, conflict_gate=0.65)))
    for g in [0.25, 0.45, 0.65, 0.85, 1.01]:  # 1.01 = always invoke
        configs.append(("gate", f"{g:.2f}", VerifierConfig(restriction="llm", gate_enabled=(g <= 1.0), conflict_gate=g)))
    for ns_variant in [3, 5, 8]:
        configs.append(("n_samples", str(ns_variant), VerifierConfig(restriction="llm", gate_enabled=True, conflict_gate=0.65)))

    # Pre-generate max(N=8) samples per example once.
    from baselines.solver import build_context_from_passages, qa_prompt
    from llm.base import extract_short_answer

    bundles: list = []
    all_shared: list = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        b = _retrieval_bundle(ex.question, ex, bm25, dense)
        bundles.append(b)
        ctx = build_context_from_passages(b.dense_passages, max_chars=1600)
        prompt = qa_prompt(ex.question, ctx)
        gens = llm.generate_n(prompt, n=max(args.n_samples, 8), seed=args.seed + i, max_new_tokens=48)
        all_shared.append([extract_short_answer(g.text).strip() for g in gens])
        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(examples) - i - 1)
            print(f"  [sample {i+1}/{len(examples)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"[sampling done] {time.time()-t0:.0f}s", flush=True)

    results = []
    for (family, label, cfg) in configs:
        verifier = SheafVerifier(encoder, cfg)
        # Choose how many samples to use
        if family == "n_samples":
            ns = int(label)
        else:
            ns = 5
        row = {"family": family, "label": label, "n_samples_used": ns}

        hits = []
        grounded = []
        latencies = []
        used_sheaf_flags = []
        for ex, bundle, shared in zip(examples, bundles, all_shared):
            cands = shared[:ns]
            res = _sheaf_pick(ex.question, bundle, cands, verifier)
            ans = res.chosen_answer
            sc = score_answer(ans, ex.answer, ex.answer_aliases, ex.dataset)
            used_titles = [p.title for p in bundle.dense_passages] + \
                          [p.title for p in bundle.bm25_passages if p.title not in {q.title for q in bundle.dense_passages}]
            gh = grounded_hit(used_titles, ex.supporting_titles, sc["hit"])
            hits.append(sc["hit"])
            grounded.append(gh)
            latencies.append(res.latency_ms)
            used_sheaf_flags.append(int(res.used_sheaf))

        row.update({
            "hit_mean": float(np.mean(hits)),
            "hit_ci95": list(bootstrap_ci(hits)),
            "grounded_mean": float(np.mean(grounded)),
            "grounded_ci95": list(bootstrap_ci(grounded)),
            "verifier_latency_ms_mean": float(np.mean(latencies)),
            "verifier_latency_ms_p95": float(np.percentile(latencies, 95)),
            "sheaf_invocation_rate": float(np.mean(used_sheaf_flags)),
        })
        results.append(row)
        print(f"  ablation family={family} label={label} hit={row['hit_mean']:.3f} grounded={row['grounded_mean']:.3f}", flush=True)

    tag = f"ablations_{slug(args.model)}_{args.dataset}_n{args.n}"
    dump_json(OUTPUTS_DIR / "ablations" / f"{tag}.json", {
        "model": args.model, "dataset": args.dataset, "n": args.n, "seed": args.seed,
        "results": results,
    })
    print(f"[done] wrote {tag}.json", flush=True)


if __name__ == "__main__":
    main()
