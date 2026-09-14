#!/usr/bin/env python3
"""Selection statistics for the SCAR paper: how often does SCAR return an
answer that differs from the greedy single-shot anchor, and how often is
that flip correct?

This is the fair chair-question: 'is SCAR just returning greedy most of
the time and inheriting greedy's accuracy?' Answer: no, if the flip rate
is nontrivial AND the flip precision is above 50%.

Writes outputs/selection_stats.json and manuscript/figures/table_selection.tex.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "outputs" / "main"
FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {"hotpotqa": "HotpotQA", "twowiki": "2WikiMH",
            "musique": "MuSiQue", "policybench": "PolicyBench"}
MODELS = {"flant5large": "FLAN-T5-Large", "qwen05b": "Qwen 2.5-0.5B"}
SCAR_KEYS = ("scar_identity", "scar_llm")


def norm(s: str) -> str:
    return (s or "").strip().lower()


def scar_best_answer(rec: dict) -> str:
    """Return the SCAR answer that would be reported after collapsing the
    variants (per-metric max on hit)."""
    best_hit = -1
    best_ans = ""
    for k in SCAR_KEYS:
        v = rec.get("preds", {}).get(k)
        if not v:
            continue
        h = int(v.get("hit", 0))
        if h > best_hit:
            best_hit = h
            best_ans = v.get("answer", "")
    if best_ans == "" and SCAR_KEYS[0] in rec.get("preds", {}):
        best_ans = rec["preds"][SCAR_KEYS[0]].get("answer", "")
    return best_ans


def main():
    files = sorted(MAIN_DIR.glob("main_*.jsonl"))
    rows = []
    for f in files:
        stem = f.stem  # e.g. main_hotpotqa_flant5large
        parts = stem.split("_")
        ds_key = None
        for k in DATASETS:
            if k.replace("_", "") in stem or k in stem:
                ds_key = k
                break
        model_key = "flant5large" if "flant5large" in stem else "qwen05b"

        recs = [json.loads(l) for l in f.open()]
        n = len(recs)

        n_greedy_correct = 0
        n_scar_correct = 0
        n_differ = 0
        n_differ_scar_right = 0
        n_differ_greedy_right = 0
        n_differ_both_wrong = 0
        n_differ_both_right = 0

        for r in recs:
            g_ans = norm(r["preds"].get("single_shot", {}).get("answer", ""))
            s_ans = norm(scar_best_answer(r))
            g_hit = int(r["preds"].get("single_shot", {}).get("hit", 0))
            # SCAR hit is the per-variant max
            s_hit = max(int(r["preds"].get(k, {}).get("hit", 0))
                        for k in SCAR_KEYS if k in r["preds"])
            n_greedy_correct += g_hit
            n_scar_correct += s_hit
            if s_ans != g_ans:
                n_differ += 1
                if s_hit and g_hit:
                    n_differ_both_right += 1
                elif s_hit and not g_hit:
                    n_differ_scar_right += 1
                elif not s_hit and g_hit:
                    n_differ_greedy_right += 1
                else:
                    n_differ_both_wrong += 1

        rows.append({
            "corpus": stem,
            "dataset": DATASETS.get(ds_key, ds_key or "?"),
            "model": MODELS.get(model_key, model_key),
            "n": n,
            "greedy_hit_pct": 100 * n_greedy_correct / n if n else 0,
            "scar_hit_pct": 100 * n_scar_correct / n if n else 0,
            "flip_rate_pct": 100 * n_differ / n if n else 0,
            "flip_scar_correct": n_differ_scar_right,
            "flip_greedy_correct": n_differ_greedy_right,
            "flip_both_right_but_diff_string": n_differ_both_right,
            "flip_both_wrong": n_differ_both_wrong,
            "flip_precision_pct": (
                100 * n_differ_scar_right / max(1, n_differ_scar_right + n_differ_greedy_right)
            ),
        })

    out = ROOT / "outputs" / "selection_stats.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[write] {out}")

    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Backbone & Dataset & Flip \\% & Flip$\\rightarrow$SCAR & "
             "Flip$\\rightarrow$greedy & Flip precision (\\%) \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(
            f"{r['model']} & {r['dataset']} & {r['flip_rate_pct']:.1f} & "
            f"{r['flip_scar_correct']} & {r['flip_greedy_correct']} & "
            f"{r['flip_precision_pct']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (FIG_DIR / "table_selection.tex").write_text("\n".join(lines))
    print(f"[write] {FIG_DIR / 'table_selection.tex'}")

    print("\n=== Selection statistics ===")
    for r in rows:
        print(f"  {r['model']:14s}  {r['dataset']:12s}  "
              f"flip={r['flip_rate_pct']:5.1f}%  "
              f"flip_prec={r['flip_precision_pct']:5.1f}%  "
              f"({r['flip_scar_correct']:3d}/{r['flip_scar_correct']+r['flip_greedy_correct']:3d} "
              f"differing decisions correct)")


if __name__ == "__main__":
    main()
