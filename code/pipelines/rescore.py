#!/usr/bin/env python3
"""Re-score SCAR variants on an existing main run's JSONL.

Reads the saved records (which contain shared_candidates), re-runs retrieval
(deterministic) and the sheaf verifier with the current code, and overwrites
the scar_* method entries in place. This lets us iterate on the verifier
without re-running expensive LLM sampling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from common import OUTPUTS_DIR, dump_json, set_all_seeds
from data_loaders.registry import load_examples
from pipelines.harness import _retrieval_bundle, _scar_reuse, aggregate, build_verifiers, grounded_hit, score_answer
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever


def rescore_file(path: Path, dataset: str, seed: int, n: int, conflict_rate: float = 0.0):
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    verifiers = build_verifiers(encoder, gate_enabled=True)
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    print(f"[load] {dataset} n={n} cr={conflict_rate}", flush=True)
    examples = load_examples(dataset, n=n, seed=seed, conflict_rate=conflict_rate)
    ex_by_qid = {ex.qid: ex for ex in examples}

    records = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    print(f"[read] {len(records)} records", flush=True)

    updated = 0
    t0 = time.time()
    for r in records:
        ex = ex_by_qid.get(r["qid"])
        if ex is None:
            continue
        cands = r["shared_candidates"]
        bundle = _retrieval_bundle(ex.question, ex, bm25, dense)
        sampling_ms = r.get("sampling_ms", 0.0)
        anchor = r["preds"].get("single_shot", {}).get("answer")
        for variant in ("identity", "linear", "llm"):
            key = f"scar_{variant}"
            if key not in r["preds"]:
                continue
            pred = _scar_reuse(ex.question, bundle, verifiers[variant], cands,
                               r["preds"][key].get("tokens", 0), sampling_ms, method_tag=key,
                               anchor=anchor)
            sc = score_answer(pred.answer, ex.answer, ex.answer_aliases, ex.dataset)
            gh = grounded_hit(pred.used_evidence_titles, ex.supporting_titles, sc["hit"])
            r["preds"][key] = {
                "answer": pred.answer,
                "candidates": pred.candidates,
                "used_titles": pred.used_evidence_titles,
                "used_sheaf": pred.used_sheaf,
                "tokens": pred.tokens,
                "latency_ms": pred.latency_ms,
                "em": sc["em"], "f1": sc["f1"], "loose_em": sc["loose_em"], "hit": sc["hit"],
                "grounded_hit": gh,
                "diagnostics": pred.diagnostics,
            }
        updated += 1
        if updated % 100 == 0:
            print(f"  rescored {updated}", flush=True)
    print(f"[done] rescored {updated} in {time.time()-t0:.0f}s", flush=True)

    # Overwrite JSONL and rewrite summary.
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    methods = list(records[0]["preds"].keys())
    summary = aggregate(records, methods)
    # Preserve the model tag from the pre-existing summary if it exists.
    old_summary_path = path.with_name(path.stem + "_summary.json")
    old_model = None
    if old_summary_path.exists():
        try:
            old_model = json.loads(old_summary_path.read_text()).get("model")
        except Exception:
            pass
    summary_meta = {
        "model": old_model,
        "dataset": dataset,
        "n_actual": len(records),
        "conflict_rate": conflict_rate,
        "results": summary,
        "rescored": True,
    }
    dump_json(old_summary_path, summary_meta)
    print(json.dumps({m: {"hit": v["hit_mean"], "em": v["em_mean"], "f1": v["f1_mean"]}
                      for m, v in summary["methods"].items()}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conflict-rate", type=float, default=0.0)
    args = ap.parse_args()
    set_all_seeds(args.seed)
    rescore_file(args.jsonl, args.dataset, args.seed, args.n, args.conflict_rate)


if __name__ == "__main__":
    main()
