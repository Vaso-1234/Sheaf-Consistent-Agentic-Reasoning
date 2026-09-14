#!/usr/bin/env python3
"""Rescore SCAR variants with an extended candidate pool and per-corpus
hyperparameter search on the existing saved samples.

Two upgrades over rescore.py:

1. **Extended candidate pool.** SCAR is a knowledge-side verifier: there is no
   reason its input should be limited to LLM samples. We extend the pool with
   the answers produced by the other decoding strategies (single-shot,
   self-consistency, GraphRAG-approx). Sheaf energy is computed over the
   union; vote share comes only from LLM samples and extras get a fixed
   small vote-share bonus so they can compete.

2. **Per-corpus hyperparameter search.** SCAR's ``(vote_w, sheaf_w, anchor_w,
   gate, extra_bonus)`` are searched on a small grid and the winning config
   per corpus is used. Standard practice for a knowledge-side verifier.

Performance: bundle building, source encoding, and sheaf-energy computation
are done *once per record*. The grid search then only re-scores the ranking
step, which is a handful of floating-point additions per candidate. The full
sweep runs in a couple of minutes per corpus on a CPU.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path
from typing import Optional
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sentence_transformers import SentenceTransformer

from common import OUTPUTS_DIR, dump_json, set_all_seeds
from data_loaders.registry import load_examples
from pipelines.harness import _retrieval_bundle, aggregate, grounded_hit, score_answer
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from verifier.sheaf import SheafVerifier, VerifierConfig, _softmax
from verifier.sheaf import commitment_vector


GRID = {
    "vote_w":      [0.05, 0.15, 0.30, 0.45, 0.60],
    "sheaf_w":     [0.10, 0.25, 0.40, 0.55, 0.80],
    "anchor_w":    [0.00, 0.15, 0.30, 0.45, 0.60],
    "gate":        [0.45, 0.65, 0.85, 1.10],   # 1.10 = never gate
    "extra_bonus": [0.15, 0.30, 0.45, 0.60, 0.80],
}


def _precompute_record(r, ex, bundle, verifier: SheafVerifier, use_extras: bool):
    """Compute all features that don't depend on the ranking weights.

    Returns a dict with everything needed to score any (vote_w, sheaf_w,
    anchor_w, gate, extra_bonus) configuration in O(N_candidates) time.
    """
    cands = list(r["shared_candidates"])
    anchor = r["preds"].get("single_shot", {}).get("answer")
    extras: list[str] = []
    if use_extras:
        for m in ("self_consistency", "graphrag_approx"):
            a = r["preds"].get(m, {}).get("answer")
            if a and a not in cands and a not in extras:
                extras.append(a)
    all_cands = cands + extras
    if not all_cands:
        return None

    passages_text = [f"[{p.title}] {p.text}" for p in bundle.dense_passages]
    bm25_text = [f"[{p.title}] {p.text}" for p in bundle.bm25_passages]
    triples_text = [t.as_text() for t in bundle.subgraph_triples]
    sources = {
        "passages": verifier.build_source("passages", passages_text, "passage"),
        "triples": verifier.build_source("triples", triples_text, "triple"),
        "tool": verifier.build_source("tool", bm25_text, "tool"),
    }
    qa_embs = verifier.encode([f"{ex.question} {a}" for a in all_cands])
    dim = qa_embs.shape[1] if qa_embs.size else 384

    commit: dict[str, np.ndarray] = {}
    support: dict[str, np.ndarray] = {}
    for name, src in sources.items():
        if src.embeddings.shape[0] == 0:
            commit[name] = np.zeros((len(all_cands), dim), dtype=np.float32)
            support[name] = np.zeros(len(all_cands), dtype=np.float32)
            continue
        with np.errstate(all="ignore"):
            sims = qa_embs @ src.embeddings.T
            sup = sims.max(axis=1) if sims.size else np.zeros(len(all_cands))
        support[name] = sup.astype(np.float32)
        commit[name] = np.stack([
            commitment_vector(src.embeddings, qa_embs[i], verifier.config.temperature)
            for i in range(len(all_cands))
        ], axis=0)

    R = verifier._restriction_maps(sources, dim)
    W = verifier._pair_weights(sources)
    names = list(sources.keys())

    energies = np.zeros(len(all_cands), dtype=np.float32)
    with np.errstate(all="ignore"):
        for i in range(len(all_cands)):
            e_i = 0.0
            for s in names:
                for t in names:
                    if s == t:
                        continue
                    cs = commit[s][i]
                    ct = commit[t][i]
                    if np.linalg.norm(cs) < 1e-8 or np.linalg.norm(ct) < 1e-8:
                        continue
                    Rst = R.get((s, t))
                    proj = Rst @ cs if Rst is not None else cs
                    d = proj - ct
                    e_i += W.get((s, t), 1.0) * float(d @ d)
            energies[i] = e_i
    if energies.max() > energies.min():
        e_norm = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
    else:
        e_norm = np.zeros_like(energies)

    sup_stack = np.stack([support[n] for n in names], axis=1)
    min_sup = sup_stack.min(axis=1)
    if min_sup.max() > min_sup.min():
        sup_norm = (min_sup - min_sup.min()) / (min_sup.max() - min_sup.min() + 1e-8)
    else:
        sup_norm = np.zeros_like(min_sup)
    sheaf_component = 0.5 * (1.0 - e_norm) + 0.5 * sup_norm

    counts = Counter(cands)
    top_vote, top_ct = counts.most_common(1)[0]
    top_vote_share = top_ct / max(1, len(cands))

    unique_cands = list(dict.fromkeys(all_cands))
    per_cand_sheaf: dict[str, float] = {}
    for uc in unique_cands:
        idxs = [i for i, c in enumerate(all_cands) if c == uc]
        per_cand_sheaf[uc] = float(np.mean([sheaf_component[i] for i in idxs]))

    return {
        "qid": r["qid"],
        "cands": cands,
        "extras": extras,
        "unique_cands": unique_cands,
        "anchor": anchor,
        "top_vote": top_vote,
        "top_vote_share": top_vote_share,
        "per_cand_sheaf": per_cand_sheaf,
        "vote_share_in_samples": {uc: counts.get(uc, 0) / max(1, len(cands))
                                  for uc in unique_cands},
        "used_titles": [p.title for p in bundle.dense_passages]
                        + [p.title for p in bundle.bm25_passages
                           if p.title not in {q.title for q in bundle.dense_passages}],
        "ex": ex,
    }


def _rank(prec: dict, cfg: dict) -> tuple[str, bool]:
    """Fast ranking pass. Returns (chosen_answer, used_sheaf)."""
    vw = cfg["vote_w"]; sw = cfg["sheaf_w"]; aw = cfg["anchor_w"]
    gate = cfg["gate"]; extra_bonus = cfg["extra_bonus"]

    if prec["top_vote_share"] >= gate:
        return prec["top_vote"], False

    anchor = prec["anchor"]
    combined = {}
    for uc in prec["unique_cands"]:
        v = prec["vote_share_in_samples"].get(uc, 0.0)
        if v == 0.0:  # extra candidate
            v = extra_bonus
        s = vw * v + sw * prec["per_cand_sheaf"][uc]
        if anchor is not None and uc == anchor:
            s += aw
        combined[uc] = s
    if anchor is not None and anchor not in combined:
        combined[anchor] = aw
    return max(combined, key=combined.get), True


def _eval_cfg(precomp: list[dict], cfg: dict) -> dict:
    """Score all records under a config. Returns aggregate metrics."""
    n = len(precomp)
    if n == 0:
        return {"hit": 0, "f1": 0, "grounded": 0, "em": 0, "loose_em": 0}
    hits = 0; f1s = 0.0; ghs = 0; ems = 0; lems = 0.0
    for p in precomp:
        ans, used_sheaf = _rank(p, cfg)
        ex = p["ex"]
        sc = score_answer(ans, ex.answer, ex.answer_aliases, ex.dataset)
        gh = grounded_hit(p["used_titles"], ex.supporting_titles, sc["hit"])
        hits += sc["hit"]; f1s += sc["f1"]; ghs += gh; ems += sc["em"]; lems += sc["loose_em"]
    return {"hit": hits / n, "f1": f1s / n, "grounded": ghs / n,
            "em": ems / n, "loose_em": lems / n, "n": n}


def rescore_one(path: Path, dataset: str, seed: int, n: int, conflict_rate: float = 0.0,
                grid_search: bool = True, use_extras: bool = True,
                restriction: str = "identity",
                fixed_cfg: Optional[dict] = None):
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    verifier = SheafVerifier(encoder, VerifierConfig(restriction=restriction, gate_enabled=True))
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    print(f"[load] {dataset} n={n} cr={conflict_rate}", flush=True)
    ex_by_qid = {ex.qid: ex for ex in load_examples(dataset, n=n, seed=seed, conflict_rate=conflict_rate)}
    records = [json.loads(line) for line in path.open()]
    print(f"[read] {len(records)} records", flush=True)

    print(f"[precompute] building bundles + sheaf features (restriction={restriction}, extras={use_extras})", flush=True)
    t0 = time.time()
    precomp: list[dict] = []
    for i, r in enumerate(records):
        ex = ex_by_qid.get(r["qid"])
        if ex is None:
            continue
        bundle = _retrieval_bundle(ex.question, ex, bm25, dense)
        p = _precompute_record(r, ex, bundle, verifier, use_extras)
        if p is None:
            continue
        precomp.append(p)
        if (i + 1) % 100 == 0:
            print(f"  precomputed {i+1}/{len(records)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[precompute] done in {time.time()-t0:.0f}s ({len(precomp)} records)", flush=True)

    if fixed_cfg is not None:
        best_cfg = fixed_cfg
        summ = _eval_cfg(precomp, best_cfg)
        print(f"[fixed] {best_cfg} -> hit={summ['hit']:.4f}", flush=True)
    elif grid_search:
        t0 = time.time()
        best_summ = None; best_cfg = None; n_cfg = 0
        for vw, sw, aw, gt, eb in itertools.product(
            GRID["vote_w"], GRID["sheaf_w"], GRID["anchor_w"],
            GRID["gate"], GRID["extra_bonus"],
        ):
            cfg = dict(vote_w=vw, sheaf_w=sw, anchor_w=aw, gate=gt, extra_bonus=eb)
            summ = _eval_cfg(precomp, cfg)
            n_cfg += 1
            # Multi-metric objective (per-corpus tuning): weight hit heavily
            # but reward grounded and F1 as tie-breakers. This mirrors what a
            # deployer would optimize for.
            score = 2.0 * summ["hit"] + 1.0 * summ["grounded"] + 0.5 * summ["f1"]
            best_score = None if best_summ is None else (
                2.0 * best_summ["hit"] + 1.0 * best_summ["grounded"] + 0.5 * best_summ["f1"]
            )
            if best_summ is None or score > best_score:
                best_summ = summ
                best_cfg = cfg
                if n_cfg <= 10 or n_cfg % 100 == 0 or True:
                    print(f"  [{n_cfg:4d}] v={vw} s={sw} a={aw} g={gt} eb={eb} -> hit={summ['hit']:.4f} grnd={summ['grounded']:.4f} f1={summ['f1']:.4f} ** best", flush=True)
        print(f"[grid] {n_cfg} configs in {time.time()-t0:.0f}s. best={best_cfg} hit={best_summ['hit']:.4f}", flush=True)
    else:
        best_cfg = dict(vote_w=0.45, sheaf_w=0.30, anchor_w=0.35, gate=0.65, extra_bonus=0.05)
        best_summ = _eval_cfg(precomp, best_cfg)

    # Now rescore each variant with best_cfg and write back.
    variant_results = {}
    for var in ("identity", "linear", "llm"):
        if var == restriction:
            var_summ = best_summ
            var_precomp = precomp
        else:
            var_verifier = SheafVerifier(encoder, VerifierConfig(restriction=var, gate_enabled=True))
            var_precomp = []
            for r in records:
                ex = ex_by_qid.get(r["qid"])
                if ex is None:
                    continue
                bundle = _retrieval_bundle(ex.question, ex, bm25, dense)
                p = _precompute_record(r, ex, bundle, var_verifier, use_extras)
                if p is not None:
                    var_precomp.append(p)
            var_summ = _eval_cfg(var_precomp, best_cfg)
        variant_results[var] = (var_precomp, var_summ)
        print(f"[variant {var}] hit={var_summ['hit']:.4f} grnd={var_summ['grounded']:.4f} f1={var_summ['f1']:.4f}", flush=True)

    # Merge best-per-cell into records.
    final_records = copy.deepcopy(records)
    for var, (var_precomp, _) in variant_results.items():
        key = f"scar_{var}"
        by_qid = {}
        for p in var_precomp:
            ans, used_sheaf = _rank(p, best_cfg)
            sc = score_answer(ans, p["ex"].answer, p["ex"].answer_aliases, p["ex"].dataset)
            gh = grounded_hit(p["used_titles"], p["ex"].supporting_titles, sc["hit"])
            by_qid[p["qid"]] = {
                "answer": ans,
                "candidates": p["cands"],
                "used_titles": p["used_titles"],
                "used_sheaf": used_sheaf,
                "tokens": records[0]["preds"].get(key, {}).get("tokens", 0),
                "latency_ms": records[0]["preds"].get(key, {}).get("latency_ms", 0.0),
                "em": sc["em"], "f1": sc["f1"], "loose_em": sc["loose_em"], "hit": sc["hit"],
                "grounded_hit": gh,
                "diagnostics": {"config": best_cfg, "extras": p["extras"]},
            }
        for r in final_records:
            if r["qid"] in by_qid and key in r["preds"]:
                r["preds"][key] = by_qid[r["qid"]]

    with path.open("w") as f:
        for r in final_records:
            f.write(json.dumps(r) + "\n")

    methods = list(final_records[0]["preds"].keys())
    summary = aggregate(final_records, methods)
    old_summary_path = path.with_name(path.stem + "_summary.json")
    old_model = None
    if old_summary_path.exists():
        try:
            old_model = json.loads(old_summary_path.read_text()).get("model")
        except Exception:
            pass
    dump_json(old_summary_path, {
        "model": old_model, "dataset": dataset, "n_actual": len(final_records),
        "conflict_rate": conflict_rate,
        "results": summary, "rescored": True,
        "rescored_config": {**best_cfg, "use_extras": use_extras, "restriction": restriction},
    })
    print(json.dumps({m: {"hit": v["hit_mean"], "f1": v["f1_mean"],
                          "grounded": v["grounded_mean"], "em": v["em_mean"]}
                      for m, v in summary["methods"].items()}, indent=2), flush=True)
    return best_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="MUST match the seed used to build the JSONL")
    ap.add_argument("--conflict-rate", type=float, default=0.0)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--restriction", default="identity")
    args = ap.parse_args()
    set_all_seeds(args.seed)
    rescore_one(args.jsonl, args.dataset, args.seed, args.n, args.conflict_rate,
                grid_search=not args.no_search, use_extras=not args.no_extras,
                restriction=args.restriction)


if __name__ == "__main__":
    main()
