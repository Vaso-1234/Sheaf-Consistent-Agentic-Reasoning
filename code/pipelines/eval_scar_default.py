#!/usr/bin/env python3
"""Evaluate SCAR under three ablation-friendly configurations:
  * default         : pre-declared tuple (0.45, 0.30, 0.35, 0.65, 0.30), extras on
  * default_noextras: same tuple with extended candidate pool disabled
  * tuned_noextras  : per-corpus tuple with extended candidate pool disabled

Writes a small JSON to outputs/ablations/scar_variants/<jsonl_stem>.json so
the manuscript can quote real 'default vs tuned', 'with extras vs without'
deltas. Does NOT touch the main summary files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from common import OUTPUTS_DIR, set_all_seeds
from data_loaders.registry import load_examples
from pipelines.harness import _retrieval_bundle
from pipelines.rescore_v2 import GRID, _eval_cfg, _precompute_record
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from verifier.sheaf import SheafVerifier, VerifierConfig


DEFAULT_CFG = dict(vote_w=0.45, sheaf_w=0.30, anchor_w=0.35,
                   gate=0.65, extra_bonus=0.30)


def _grid_best(precomp, obj):
    """Return (best_summ, best_cfg) after searching the pre-declared grid."""
    best_summ = None
    best_cfg = None
    for vw, sw, aw, gt, eb in itertools.product(
        GRID["vote_w"], GRID["sheaf_w"], GRID["anchor_w"],
        GRID["gate"], GRID["extra_bonus"],
    ):
        cfg = dict(vote_w=vw, sheaf_w=sw, anchor_w=aw, gate=gt, extra_bonus=eb)
        summ = _eval_cfg(precomp, cfg)
        score = obj(summ)
        if best_summ is None or score > obj(best_summ):
            best_summ = summ
            best_cfg = cfg
    return best_summ, best_cfg


def _precompute(records, ex_by_qid, verifier, bm25, dense, use_extras):
    precomp = []
    t0 = time.time()
    for i, r in enumerate(records):
        ex = ex_by_qid.get(r["qid"])
        if ex is None:
            continue
        bundle = _retrieval_bundle(ex.question, ex, bm25, dense)
        p = _precompute_record(r, ex, bundle, verifier, use_extras=use_extras)
        if p is not None:
            precomp.append(p)
        if (i + 1) % 100 == 0:
            print(f"  precomputed {i+1}/{len(records)} extras={use_extras} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return precomp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conflict-rate", type=float, default=0.0)
    ap.add_argument("--restriction", default="identity")
    ap.add_argument("--out-dir", default="outputs/ablations/scar_variants")
    args = ap.parse_args()
    set_all_seeds(args.seed)

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    verifier = SheafVerifier(
        encoder,
        VerifierConfig(restriction=args.restriction, gate_enabled=True),
    )
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    ex_by_qid = {
        ex.qid: ex
        for ex in load_examples(args.dataset, n=args.n, seed=args.seed,
                                conflict_rate=args.conflict_rate)
    }
    records = [json.loads(line) for line in args.jsonl.open()]
    print(f"[load] dataset={args.dataset} n={args.n} cr={args.conflict_rate} "
          f"records={len(records)}", flush=True)

    obj = lambda s: 2.0 * s["hit"] + 1.0 * s["grounded"] + 0.5 * s["f1"]

    print("[precompute] extras=True", flush=True)
    precomp_ext = _precompute(records, ex_by_qid, verifier, bm25, dense,
                              use_extras=True)
    print(f"[precompute] extras=True done ({len(precomp_ext)} records)",
          flush=True)

    print("[precompute] extras=False", flush=True)
    precomp_noext = _precompute(records, ex_by_qid, verifier, bm25, dense,
                                use_extras=False)
    print(f"[precompute] extras=False done ({len(precomp_noext)} records)",
          flush=True)

    result = {
        "dataset": args.dataset,
        "conflict_rate": args.conflict_rate,
        "seed": args.seed,
        "n": len(precomp_ext),
        "restriction": args.restriction,
        "variants": {},
    }

    print("[eval] default cfg, extras=True", flush=True)
    result["variants"]["default_extras"] = {
        "cfg": DEFAULT_CFG, "use_extras": True,
        "summ": _eval_cfg(precomp_ext, DEFAULT_CFG),
    }
    print("[eval] default cfg, extras=False", flush=True)
    result["variants"]["default_noextras"] = {
        "cfg": DEFAULT_CFG, "use_extras": False,
        "summ": _eval_cfg(precomp_noext, DEFAULT_CFG),
    }
    print("[eval] grid search, extras=True", flush=True)
    tuned_ext_summ, tuned_ext_cfg = _grid_best(precomp_ext, obj)
    result["variants"]["tuned_extras"] = {
        "cfg": tuned_ext_cfg, "use_extras": True, "summ": tuned_ext_summ,
    }
    print("[eval] grid search, extras=False", flush=True)
    tuned_noext_summ, tuned_noext_cfg = _grid_best(precomp_noext, obj)
    result["variants"]["tuned_noextras"] = {
        "cfg": tuned_noext_cfg, "use_extras": False, "summ": tuned_noext_summ,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.jsonl.stem + "_ablation.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[write] {out_path}", flush=True)
    print(json.dumps({k: v["summ"] for k, v in result["variants"].items()},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
