#!/usr/bin/env python3
"""Conflict-injection experiment: PolicyBench at multiple conflict rates.

The paper's key controlled experiment. Only PolicyBench supports programmatic
conflict injection (numerically wrong paraphrases). We report accuracy vs
conflict rate for the primary model.
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
from llm.registry import build_llm
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from pipelines.harness import METHOD_LIST_DEFAULT, aggregate, build_verifiers, run_one_example


def slug(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="policybench")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conflict-rates", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--methods", nargs="+", default=METHOD_LIST_DEFAULT)
    ap.add_argument("--skip-heavy", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    methods = list(args.methods)
    if args.skip_heavy and "process_reward" in methods:
        methods.remove("process_reward")

    print(f"[load] LLM={args.model}", flush=True)
    llm = build_llm(args.model, max_new_tokens=48)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    verifiers = build_verifiers(encoder, gate_enabled=True)

    all_summaries = {}
    for cr in args.conflict_rates:
        print(f"\n===== conflict_rate = {cr} =====", flush=True)
        examples = load_examples(args.dataset, n=args.n, seed=args.seed, conflict_rate=cr)
        t0 = time.time()
        records = []
        tag = f"conflict_{slug(args.model)}_cr{int(cr*100):02d}"
        out_path = OUTPUTS_DIR / "conflict" / f"{tag}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for i, ex in enumerate(examples):
                try:
                    rec = run_one_example(ex, llm, bm25, dense, verifiers, methods,
                                          n_samples=args.n_samples, seed=args.seed + i,
                                          max_new_tokens=48)
                    records.append(rec)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"[warn] q{i} failed: {e}", flush=True)
                if (i + 1) % args.log_every == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (len(examples) - i - 1)
                    print(f"  [{i+1}/{len(examples)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        summary = aggregate(records, methods)
        meta = {
            "model": args.model,
            "dataset": args.dataset,
            "conflict_rate": cr,
            "n_actual": len(records),
            "methods": methods,
            "wall_seconds": time.time() - t0,
            "results": summary,
        }
        dump_json(OUTPUTS_DIR / "conflict" / f"{tag}_summary.json", meta)
        all_summaries[str(cr)] = meta
        print(json.dumps({m: v["hit_mean"] for m, v in summary["methods"].items()}, indent=2), flush=True)

    dump_json(OUTPUTS_DIR / "conflict" / f"conflict_sweep_{slug(args.model)}.json", all_summaries)
    print("\n[done] wrote conflict sweep summary", flush=True)


if __name__ == "__main__":
    main()
