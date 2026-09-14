#!/usr/bin/env python3
"""Micro-benchmark the CPU-side sheaf verifier over many synthetic calls."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sentence_transformers import SentenceTransformer

from common import OUTPUTS_DIR, dump_json
from verifier.sheaf import SheafVerifier, VerifierConfig


def synthetic_case(rng: random.Random, k_p: int = 5, k_g: int = 8, k_t: int = 5, n_cands: int = 5) -> dict:
    cands = ["Tim Burton", "Ed Wood", "Steven Spielberg", "Martin Scorsese", "Wes Anderson"][:n_cands]
    passages = [f"Passage {i}: entity {rng.choice(['Alpha','Beta','Gamma'])} interacts with concept {rng.choice(['X','Y','Z'])}." for i in range(k_p)]
    triples = [f"E{i} | relates_to | E{(i+3)%k_g}" for i in range(k_g)]
    tool = [f"Tool observation {i}: {rng.choice(cands)} appears near evidence {i}." for i in range(k_t)]
    return {"cands": cands, "p": passages, "g": triples, "t": tool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    verifiers = {
        r: SheafVerifier(encoder, VerifierConfig(restriction=r, gate_enabled=False))
        for r in ("identity", "linear", "llm")
    }

    rng = random.Random(args.seed)
    results = {}
    for variant, ver in verifiers.items():
        latencies = []
        for i in range(args.calls):
            c = synthetic_case(rng, k_p=rng.randint(3, 7), k_g=rng.randint(4, 10),
                                k_t=rng.randint(3, 7))
            src = {
                "passages": ver.build_source("passages", c["p"], "passage"),
                "triples": ver.build_source("triples", c["g"], "triple"),
                "tool": ver.build_source("tool", c["t"], "tool"),
            }
            t0 = time.time()
            res = ver.verify("query", c["cands"], src)
            # We report the "verifier only" component, i.e. the value returned by
            # verify() itself (which excludes any external encoding).
            latencies.append(res.latency_ms)
            if (i + 1) % 500 == 0:
                print(f"  {variant} [{i+1}/{args.calls}]", flush=True)
        arr = np.asarray(latencies)
        results[variant] = {
            "calls": args.calls,
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "std_ms": float(arr.std()),
        }
        print(f"[{variant}] mean={arr.mean():.2f}ms  median={np.median(arr):.2f}ms  p95={np.percentile(arr,95):.2f}ms  p99={np.percentile(arr,99):.2f}ms", flush=True)

    dump_json(OUTPUTS_DIR / "latency" / f"latency_bench_{args.calls}.json", results)
    print("[done] wrote latency bench", flush=True)


if __name__ == "__main__":
    main()
