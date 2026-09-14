#!/usr/bin/env python3
"""Main experiment: all methods on all datasets for a given model."""

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
    ap.add_argument("--model", required=True, help="e.g. google/flan-t5-large")
    ap.add_argument("--dataset", required=True,
                    choices=["hotpotqa", "hotpotqa-bridge", "hotpotqa-comparison",
                             "two_wiki", "musique", "policybench"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conflict-rate", type=float, default=0.0,
                    help="Only for policybench")
    ap.add_argument("--methods", nargs="+", default=METHOD_LIST_DEFAULT)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="Skip process_reward (which doubles LLM cost)")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    methods = list(args.methods)
    if args.skip_heavy and "process_reward" in methods:
        methods.remove("process_reward")

    print(f"[load] dataset={args.dataset} n={args.n} cr={args.conflict_rate}", flush=True)
    examples = load_examples(args.dataset, n=args.n, seed=args.seed, conflict_rate=args.conflict_rate)
    print(f"[load] got {len(examples)} examples", flush=True)

    print(f"[load] LLM={args.model}", flush=True)
    llm = build_llm(args.model, max_new_tokens=48)
    print(f"[load] encoder", flush=True)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    bm25 = BM25Retriever()
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    verifiers = build_verifiers(encoder, gate_enabled=True)

    t0 = time.time()
    records = []
    tag = args.out_tag or f"{args.dataset}_{slug(args.model)}"
    out_path = OUTPUTS_DIR / "main" / f"{tag}.jsonl"
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
                continue
            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(examples) - i - 1)
                print(f"  [{i+1}/{len(examples)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    total = time.time() - t0
    print(f"[done] {len(records)} in {total:.0f}s ({total/max(1,len(records)):.2f}s/query)", flush=True)

    summary = aggregate(records, methods)
    summary_meta = {
        "model": args.model,
        "dataset": args.dataset,
        "n_requested": args.n,
        "n_actual": len(records),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "conflict_rate": args.conflict_rate,
        "methods": methods,
        "wall_seconds": total,
        "results": summary,
    }
    dump_json(OUTPUTS_DIR / "main" / f"{tag}_summary.json", summary_meta)
    print(json.dumps({m: {"hit": v["hit_mean"], "grounded": v["grounded_mean"], "em": v["em_mean"], "f1": v["f1_mean"]}
                      for m, v in summary["methods"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
