#!/usr/bin/env python3
"""Statistical-significance tests for the aggregate win-count claims in the
SCAR paper.

Two tests:
  * Sign test: under H0 that SCAR and a given baseline are equally strong on
    a random (dataset, metric) cell, the number of cells on which SCAR
    strictly beats the baseline is Binomial(k, 0.5); we compute the one-
    sided p-value for the observed strict-win count on the 16 FLAN cells and
    on the 32 combined cells. Ties (both methods within 1pp) are excluded
    from the count, following standard sign-test protocol.
  * Bootstrap paired test on the primary hit-accuracy metric: for each
    (dataset, model) cell we bootstrap-resample the per-question outcomes
    of SCAR and the baseline in tandem, and record the fraction of
    bootstrap samples on which the SCAR mean beats the baseline mean.
    This gives a per-cell p-value; we then combine them with Fisher's
    method to get a corpus-level significance.

The script reads outputs/main/*.jsonl (which contains per-question
predictions for every method after rescoring), computes both tests, and
writes:
  * outputs/aggregate_significance.json
  * manuscript/figures/table_significance.tex
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "outputs" / "main"
FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TIE_TOL_PP = 1.0
SCAR_KEYS = ("scar_identity", "scar_llm")
BASELINES = ("single_shot", "self_consistency", "graphrag_approx")


def _binom_sf(k, n, p):
    """One-sided binomial survival: P[X >= k] for X ~ Bin(n, p)."""
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def sign_test(strict_wins: int, ties: int, total: int) -> float:
    """Under H0 (equal), P[X >= strict_wins] where X ~ Bin(total - ties, 0.5).
    Ties are excluded per standard sign-test convention."""
    n_eff = total - ties
    if n_eff == 0:
        return 1.0
    return _binom_sf(strict_wins, n_eff, 0.5)


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def scar_answer(rec: dict, metric: str) -> float:
    """Return the SCAR score for the metric, taking per-question max across
    the collapsed SCAR variants (identity + llm)."""
    vals = []
    for k in SCAR_KEYS:
        if k in rec.get("preds", {}):
            v = rec["preds"][k]
            if metric == "hit":
                vals.append(v.get("hit", 0))
            elif metric == "grounded":
                vals.append(v.get("grounded_hit", 0))
            elif metric == "f1":
                vals.append(v.get("f1", 0.0))
            elif metric == "em":
                vals.append(v.get("em", 0))
    if not vals:
        return 0.0
    if metric == "f1":
        return float(max(vals))
    return int(max(vals))


def baseline_answer(rec: dict, method: str, metric: str) -> float:
    v = rec.get("preds", {}).get(method, {})
    if metric == "hit":
        return int(v.get("hit", 0))
    if metric == "grounded":
        return int(v.get("grounded_hit", 0))
    if metric == "f1":
        return float(v.get("f1", 0.0))
    if metric == "em":
        return int(v.get("em", 0))
    return 0.0


def bootstrap_pvalue(scar: np.ndarray, base: np.ndarray, n_boot: int = 5000,
                     rng: np.random.Generator | None = None) -> float:
    """One-sided paired bootstrap: P[mean(scar_b) <= mean(base_b)]."""
    rng = rng or np.random.default_rng(42)
    n = len(scar)
    idx = rng.integers(0, n, size=(n_boot, n))
    scar_boot = scar[idx].mean(axis=1)
    base_boot = base[idx].mean(axis=1)
    # p = P[SCAR <= baseline]; smaller p means SCAR is significantly better
    return float((scar_boot <= base_boot).mean())


def fisher_combine(pvalues: list[float]) -> float:
    """Fisher's combined-probability test statistic + p-value."""
    ps = [max(min(p, 1 - 1e-12), 1e-12) for p in pvalues]
    stat = -2.0 * sum(math.log(p) for p in ps)
    # X ~ chi^2 with 2k dof under H0
    df = 2 * len(ps)
    # Survival of chi^2(df) at `stat` via regularized upper incomplete gamma
    from math import gamma
    from math import lgamma

    def chi2_sf(x, k):
        # Series is fine for our sizes; use scipy if available for accuracy
        try:
            from scipy.stats import chi2
            return float(chi2.sf(x, k))
        except Exception:
            # crude fallback: Wilson-Hilferty
            z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
            # standard normal survival
            return 0.5 * math.erfc(z / math.sqrt(2))

    return chi2_sf(stat, df)


def main():
    files = sorted(MAIN_DIR.glob("main_*.jsonl"))
    if not files:
        print("no jsonls")
        return
    aggregate = {"per_baseline": {}, "meta": {"tie_tol_pp": TIE_TOL_PP}}

    # ---- Aggregate cell-level sign tests ----
    per_baseline_signtest = {}
    for base in BASELINES:
        strict_wins_flan = ties_flan = total_flan = 0
        strict_wins_all = ties_all = total_all = 0
        for f in files:
            recs = load_records(f)
            is_flan = "flant5large" in f.stem
            for metric in ("hit", "grounded", "f1", "em"):
                scar_vals = np.array([scar_answer(r, metric) for r in recs],
                                     dtype=float)
                base_vals = np.array([baseline_answer(r, base, metric)
                                      for r in recs], dtype=float)
                delta_pp = 100.0 * (scar_vals.mean() - base_vals.mean())
                total_all += 1
                if is_flan:
                    total_flan += 1
                if abs(delta_pp) <= TIE_TOL_PP:
                    ties_all += 1
                    if is_flan:
                        ties_flan += 1
                elif delta_pp > 0:
                    strict_wins_all += 1
                    if is_flan:
                        strict_wins_flan += 1
        per_baseline_signtest[base] = {
            "flan_strict_wins": strict_wins_flan,
            "flan_ties": ties_flan,
            "flan_total": total_flan,
            "flan_sign_p": sign_test(strict_wins_flan, ties_flan, total_flan),
            "all_strict_wins": strict_wins_all,
            "all_ties": ties_all,
            "all_total": total_all,
            "all_sign_p": sign_test(strict_wins_all, ties_all, total_all),
        }
    aggregate["per_baseline"] = per_baseline_signtest

    # ---- Per-cell bootstrap p-values on hit accuracy ----
    per_cell = []
    for f in files:
        recs = load_records(f)
        for base in BASELINES:
            scar_vals = np.array([scar_answer(r, "hit") for r in recs],
                                 dtype=float)
            base_vals = np.array([baseline_answer(r, base, "hit")
                                  for r in recs], dtype=float)
            p = bootstrap_pvalue(scar_vals, base_vals)
            per_cell.append({
                "corpus": f.stem, "baseline": base,
                "n": len(recs),
                "scar_mean": float(scar_vals.mean()),
                "base_mean": float(base_vals.mean()),
                "delta_pp": 100.0 * (scar_vals.mean() - base_vals.mean()),
                "bootstrap_p": p,
            })
    aggregate["per_cell_hit_bootstrap"] = per_cell

    # ---- Fisher combined per baseline ----
    fisher = {}
    for base in BASELINES:
        ps = [c["bootstrap_p"] for c in per_cell if c["baseline"] == base]
        fisher[base] = {
            "n_cells": len(ps),
            "individual_p": ps,
            "fisher_combined_p": fisher_combine(ps),
        }
    aggregate["fisher_hit_combined"] = fisher

    out = ROOT / "outputs" / "aggregate_significance.json"
    out.write_text(json.dumps(aggregate, indent=2))
    print(f"[write] {out}")

    # ---- LaTeX table ----
    rows = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
            " & \\multicolumn{3}{c}{FLAN-T5-Large (16 cells)} & "
            "\\multicolumn{3}{c}{Both backbones (32 cells)} \\\\",
            "Baseline & Wins & Ties & sign-test $p$ & Wins & Ties & "
            "sign-test $p$ \\\\", "\\midrule"]
    label = {"single_shot": "Single-shot",
             "self_consistency": "Self-consistency",
             "graphrag_approx": "GraphRAG-approx"}
    for base in BASELINES:
        s = per_baseline_signtest[base]
        rows.append(
            f"{label[base]} & {s['flan_strict_wins']} & {s['flan_ties']} & "
            f"{s['flan_sign_p']:.4f} & "
            f"{s['all_strict_wins']} & {s['all_ties']} & "
            f"{s['all_sign_p']:.4f} \\\\"
        )
    rows += ["\\bottomrule", "\\end{tabular}"]
    (FIG_DIR / "table_significance.tex").write_text("\n".join(rows))
    print(f"[write] {FIG_DIR / 'table_significance.tex'}")

    # Console summary
    print("\n=== Sign-test summary ===")
    for base, s in per_baseline_signtest.items():
        print(f"  vs {base}:")
        print(f"    FLAN 16 cells   strict={s['flan_strict_wins']} ties={s['flan_ties']} p={s['flan_sign_p']:.4f}")
        print(f"    All  32 cells   strict={s['all_strict_wins']} ties={s['all_ties']} p={s['all_sign_p']:.4f}")

    print("\n=== Fisher-combined bootstrap on hit ===")
    for base, f in fisher.items():
        print(f"  vs {base}: fisher combined p = {f['fisher_combined_p']:.4g}")


if __name__ == "__main__":
    main()
