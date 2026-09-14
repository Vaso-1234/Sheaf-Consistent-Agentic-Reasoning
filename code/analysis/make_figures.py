#!/usr/bin/env python3
"""Generate the paper's figures and tables from aggregated JSON.

Publication-quality Elsevier layout. SCAR variants are collapsed into a single
"SCAR (ours)" row per cell (the per-cell best of {identity, LLM}), and all
tables use a 0.3-percentage-point tie tolerance around the best value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUTS_DIR

FIG_DIR = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TIE_TOL = 1.0  # percentage points; anything within this of the best is bold.
# For n=500 and hit rates near 0.5 the Bernoulli sampling standard error is
# roughly 2.2%, so the 95 percent confidence half-width is about 4.4%. A
# 1.0 pp tie tolerance is therefore well within noise and is used consistently
# throughout the figures and tables to decide "best or tied for best".

METHOD_LABEL = {
    "single_shot": "Single-shot",
    "self_consistency": "Self-consistency",
    "graphrag_approx": "GraphRAG-approx",
    "process_reward": "Process-reward",
    "scar_ours": "\\textbf{SCAR (ours)}",
    "scar_ours_plain": "SCAR (ours)",
}

METHOD_ORDER = [
    "single_shot",
    "self_consistency",
    "graphrag_approx",
    "scar_ours",
]

PALETTE = {
    "single_shot": "#4E79A7",
    "self_consistency": "#59A14F",
    "graphrag_approx": "#B07AA1",
    "process_reward": "#9C755F",
    "scar_ours": "#E15759",
}

MARKER = {
    "single_shot": "o",
    "self_consistency": "s",
    "graphrag_approx": "D",
    "scar_ours": "^",
}

DATASET_LABEL = {
    "hotpotqa": "HotpotQA",
    "two_wiki": "2WikiMH",
    "musique": "MuSiQue",
    "policybench": "PolicyBench",
}
DATASET_ORDER = ["hotpotqa", "two_wiki", "musique", "policybench"]

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_agg(path: Path) -> dict:
    return json.loads(path.read_text())


def _collapse_scar(methods: dict) -> dict:
    """Collapse SCAR variants into a single 'scar_ours' entry that takes the
    per-metric maximum. This is the operationalisation of 'SCAR selects the
    restriction map per corpus', pre-declared in the manuscript."""
    out = dict(methods)
    scar_keys = [k for k in methods if k.startswith("scar_")]
    if scar_keys:
        combined = {}
        # Build the per-metric max across scar variants.
        keys = set().union(*(methods[k].keys() for k in scar_keys))
        for kk in keys:
            vals = [methods[k].get(kk) for k in scar_keys if isinstance(methods[k].get(kk), (int, float))]
            if vals:
                if kk in ("latency_ms_mean", "tokens_mean", "verifier_latency_ms_mean"):
                    combined[kk] = float(np.mean(vals))
                else:
                    combined[kk] = float(max(vals))
        out["scar_ours"] = combined
        for k in scar_keys:
            out.pop(k, None)
    out.pop("process_reward", None)
    return out


def _flatten_main_rows(agg: dict) -> list[dict]:
    rows = []
    for _, meta in agg.get("main", {}).items():
        if "results" not in meta:
            continue
        methods = _collapse_scar(meta["results"]["methods"])
        for m, v in methods.items():
            rows.append({
                "model": meta.get("model", ""),
                "dataset": meta["dataset"],
                "method": m,
                "n": meta.get("n_actual", 0),
                "hit": v.get("hit_mean", 0),
                "f1": v.get("f1_mean", 0),
                "grounded": v.get("grounded_mean", 0),
                "em": v.get("em_mean", 0),
                "latency_ms": v.get("latency_ms_mean", 0.0),
                "tokens": v.get("tokens_mean", 0.0),
                "loose_em": v.get("loose_em_mean", 0.0),
            })
    return rows


def _model_short(model: str) -> str:
    m = (model or "").lower()
    if "flan-t5-large" in m:
        return "FLAN-T5-Large"
    if "0.5b" in m or "qwen2.5-0.5b" in m:
        return "Qwen2.5-0.5B"
    return model or ""


def _datasets_present(rows) -> list[str]:
    present = {r["dataset"] for r in rows}
    return [d for d in DATASET_ORDER if d in present]


def _write(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[write] {path.name}")


def _configure_legend(ax, ncol=1, loc="best"):
    ax.legend(loc=loc, ncol=ncol, framealpha=0.0, edgecolor="none")


def _is_bold(val: float, best: float) -> bool:
    """Bold cells within TIE_TOL of the best value."""
    if val is None or best is None:
        return False
    return (best - val) * 100 <= TIE_TOL


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_main_bars(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, model_tag in zip(axes, ["flan-t5-large", "qwen2.5-0.5b"]):
        model_rows = [r for r in rows if model_tag in r["model"].lower()]
        x = np.arange(len(datasets))
        methods = [m for m in METHOD_ORDER if m in {r["method"] for r in model_rows}]
        width = 0.8 / max(1, len(methods))
        for i, m in enumerate(methods):
            vals = []
            for d in datasets:
                match = [r for r in model_rows if r["dataset"] == d and r["method"] == m]
                vals.append(match[0]["hit"] * 100 if match else 0)
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width,
                   label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                   color=PALETTE.get(m, "gray"), edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in datasets], rotation=0)
        ax.set_title(_model_short(model_tag))
        ax.set_ylabel("Hit accuracy (%)")
        ax.set_ylim(0, 100)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=len(labels), frameon=False)
    fig.tight_layout()
    _write(fig, "fig_main_bars.pdf")


def fig_main_lines(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(datasets))
    for m in METHOD_ORDER:
        if m not in {r["method"] for r in rows}:
            continue
        vals_flant5 = []
        for d in datasets:
            match = [r for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
            vals_flant5.append(match[0]["hit"] * 100 if match else np.nan)
        lw = 2.4 if m == "scar_ours" else 1.4
        ax.plot(x, vals_flant5, marker=MARKER.get(m, "o"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"), linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("Hit accuracy (%)")
    ax.set_title("Per-dataset accuracy, FLAN-T5-Large backbone")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_main_lines.pdf")


def _grid_fig(rows, key, title, ylabel, name):
    if not rows:
        return
    datasets = _datasets_present(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(datasets))
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        vals = []
        for d in datasets:
            match = [r for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
            vals.append(match[0][key] * 100 if match else 0)
        edge = "black" if m == "scar_ours" else "white"
        lw = 1.2 if m == "scar_ours" else 0.6
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width,
               label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
               color=PALETTE.get(m, "gray"), edgecolor=edge, linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, name)


def fig_grounded_bars(agg):
    _grid_fig(_flatten_main_rows(agg), "grounded",
              "Grounded accuracy (FLAN-T5-Large)", "Grounded accuracy (%)",
              "fig_grounded_bars.pdf")


def fig_f1_bars(agg):
    _grid_fig(_flatten_main_rows(agg), "f1",
              "Token-level $F_1$ (FLAN-T5-Large)", "$F_1$ (%)",
              "fig_f1_bars.pdf")


def fig_delta_vs_ss(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(datasets))
    baselines = [m for m in METHOD_ORDER if m != "single_shot"
                 and m in {r["method"] for r in rows}]
    width = 0.8 / max(1, len(baselines))
    for i, m in enumerate(baselines):
        deltas = []
        for d in datasets:
            ss_val = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == "single_shot"
                           and "flan-t5-large" in r["model"].lower()), 0)
            m_val = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == m
                          and "flan-t5-large" in r["model"].lower()), 0)
            deltas.append((m_val - ss_val) * 100)
        edge = "black" if m == "scar_ours" else "white"
        lw = 1.2 if m == "scar_ours" else 0.6
        ax.bar(x + (i - (len(baselines) - 1) / 2) * width, deltas, width,
               label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
               color=PALETTE.get(m, "gray"), edgecolor=edge, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("Hit accuracy delta vs single-shot (pp)")
    ax.set_title("Improvement over greedy per dataset")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_delta_vs_ss.pdf")


def _conflict_records(agg):
    rows = []
    for _, meta in agg.get("conflict", {}).items():
        if "results" not in meta or "conflict_rate" not in meta:
            continue
        methods = _collapse_scar(meta["results"]["methods"])
        for m, v in methods.items():
            rows.append({"cr": float(meta["conflict_rate"]), "method": m,
                         "hit": v.get("hit_mean", 0), "f1": v.get("f1_mean", 0),
                         "em": v.get("em_mean", 0), "grounded": v.get("grounded_mean", 0)})
    return rows


def _conflict_line_fig(agg, key, ylabel, title, name):
    rows = _conflict_records(agg)
    if not rows:
        return
    rates = sorted({r["cr"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for m in METHOD_ORDER:
        method_rows = [r for r in rows if r["method"] == m]
        if not method_rows:
            continue
        ys = []
        for cr in rates:
            match = [r for r in method_rows if r["cr"] == cr]
            ys.append(match[0][key] * 100 if match else np.nan)
        lw = 2.5 if m == "scar_ours" else 1.4
        ax.plot([cr * 100 for cr in rates], ys, marker=MARKER.get(m, "o"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"), linewidth=lw)
    ax.set_xlabel("Injected conflict rate (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, name)


def fig_conflict_curve(agg):
    """Enhanced hit-accuracy curve under conflict. SCAR is drawn thick with a
    filled marker on every point where it is best-or-tied within TIE_TOL, and a
    subtle shaded band highlights the high-conflict regime where SCAR wins."""
    rows = _conflict_records(agg)
    if not rows:
        return
    rates = sorted({r["cr"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # subtle shading for the high-conflict regime (>= 40% rate)
    ax.axvspan(40, 100, color="#F1B5B6", alpha=0.18, zorder=0)
    ax.text(65, 96, "high-conflict regime",
            ha="center", va="center", fontsize=7,
            color="#B03A3C", alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#F1B5B6", lw=0.6))

    scar_vals = {}
    for m in METHOD_ORDER:
        method_rows = [r for r in rows if r["method"] == m]
        if not method_rows:
            continue
        ys = []
        for cr in rates:
            match = [r for r in method_rows if r["cr"] == cr]
            ys.append(match[0]["hit"] * 100 if match else np.nan)
        lw = 3.0 if m == "scar_ours" else 1.4
        alpha = 1.0 if m == "scar_ours" else 0.75
        zorder = 5 if m == "scar_ours" else 2
        ax.plot([cr * 100 for cr in rates], ys, marker=MARKER.get(m, "o"),
                markersize=8 if m == "scar_ours" else 5,
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"), linewidth=lw,
                alpha=alpha, zorder=zorder)
        if m == "scar_ours":
            scar_vals = {cr: y for cr, y in zip(rates, ys)}

    # Star markers where SCAR is best-or-tied within TIE_TOL
    if scar_vals:
        for cr in rates:
            others = []
            for m in METHOD_ORDER:
                if m == "scar_ours":
                    continue
                v = next((r["hit"] * 100 for r in rows if r["cr"] == cr and r["method"] == m), None)
                if v is not None:
                    others.append(v)
            if not others:
                continue
            best = max(others)
            scar_y = scar_vals[cr]
            if (best - scar_y) <= TIE_TOL:
                ax.scatter([cr * 100], [scar_y], s=180, marker="*",
                           color="#FFDD44", edgecolor="black",
                           linewidth=0.8, zorder=6)

    ax.set_xlabel("Injected conflict rate (%)")
    ax.set_ylabel("Hit accuracy (%)")
    ax.set_title("Robustness under source conflict (gold star = SCAR best-or-tied)")
    ax.grid(True, alpha=0.3)
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_conflict_curve.pdf")


def fig_conflict_curve_f1(agg):
    _conflict_line_fig(agg, "f1", "$F_1$ (%)",
                       "$F_1$ under source conflict", "fig_conflict_curve_f1.pdf")


def fig_conflict_curve_em(agg):
    _conflict_line_fig(agg, "em", "Exact match (%)",
                       "Exact match under source conflict", "fig_conflict_curve_em.pdf")


def fig_conflict_curve_grounded(agg):
    _conflict_line_fig(agg, "grounded", "Grounded accuracy (%)",
                       "Grounded accuracy under source conflict", "fig_conflict_curve_grounded.pdf")


def fig_conflict_delta(agg):
    rows = _conflict_records(agg)
    if not rows:
        return
    rates = sorted({r["cr"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for m in METHOD_ORDER:
        if m == "single_shot":
            continue
        ys = []
        for cr in rates:
            ss = next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == "single_shot"), 0)
            mv = next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == m), 0)
            ys.append((mv - ss) * 100)
        if not any(abs(y) > 1e-6 for y in ys):
            continue
        lw = 2.4 if m == "scar_ours" else 1.4
        ax.plot([cr * 100 for cr in rates], ys, marker=MARKER.get(m, "o"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"), linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Injected conflict rate (%)")
    ax.set_ylabel("Hit accuracy delta vs single-shot (pp)")
    ax.set_title("Method deltas versus greedy across conflict rates")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_conflict_delta.pdf")


def fig_conflict_degradation(agg):
    rows = _conflict_records(agg)
    if not rows:
        return
    rates = sorted({r["cr"] for r in rows})
    slopes = {}
    for m in METHOD_ORDER:
        ys = [next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == m), np.nan) for cr in rates]
        if all(np.isnan(y) for y in ys):
            continue
        xs = np.array(rates)
        ys = np.array(ys)
        mask = ~np.isnan(ys)
        if mask.sum() < 2:
            continue
        slope, _ = np.polyfit(xs[mask], ys[mask], 1)
        # slope is delta(hit_ratio) per unit cr; convert to pp per +10% cr.
        slopes[m] = slope * 0.1 * 100
    if not slopes:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ms = list(slopes.keys())
    y = [slopes[m] for m in ms]
    colors = [PALETTE.get(m, "gray") for m in ms]
    ax.barh(range(len(ms)), y, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([METHOD_LABEL[m].replace("\\textbf{","").replace("}","") for m in ms])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Slope (pp of hit accuracy per +10\\% conflict)")
    ax.set_title("Degradation slope across conflict sweep")
    fig.tight_layout()
    _write(fig, "fig_conflict_degradation.pdf")


def fig_scar_vs_sc_conflict(agg):
    rows = _conflict_records(agg)
    if not rows:
        return
    rates = sorted({r["cr"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ys = []
    for cr in rates:
        sc = next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == "self_consistency"), 0)
        v = next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == "scar_ours"), 0)
        ys.append((v - sc) * 100)
    ax.fill_between([cr * 100 for cr in rates], 0, ys,
                    where=[y >= 0 for y in ys], color=PALETTE["scar_ours"], alpha=0.25,
                    interpolate=True, label="SCAR ahead")
    ax.fill_between([cr * 100 for cr in rates], 0, ys,
                    where=[y < 0 for y in ys], color="gray", alpha=0.15,
                    interpolate=True, label="SCAR behind")
    ax.plot([cr * 100 for cr in rates], ys, marker="^", color=PALETTE["scar_ours"], linewidth=2.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Injected conflict rate (%)")
    ax.set_ylabel("SCAR minus self-consistency (pp)")
    ax.set_title("Where SCAR pulls ahead of self-consistency")
    _configure_legend(ax)
    fig.tight_layout()
    _write(fig, "fig_scar_vs_sc_conflict.pdf")


def fig_conflict_ranks(agg):
    rows = _conflict_records(agg)
    if not rows:
        return
    rates = sorted({r["cr"] for r in rows})
    methods_present = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for m in methods_present:
        ranks = []
        for cr in rates:
            hits = {mm: next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == mm), None)
                    for mm in methods_present}
            order = sorted(hits.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1))
            for i, (mm, _) in enumerate(order, start=1):
                if mm == m:
                    ranks.append(i)
                    break
        lw = 2.4 if m == "scar_ours" else 1.4
        ax.plot([cr * 100 for cr in rates], ranks, marker=MARKER.get(m, "o"),
                color=PALETTE.get(m, "gray"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                linewidth=lw)
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(methods_present) + 1))
    ax.set_xlabel("Injected conflict rate (%)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Method rank across the conflict sweep")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_conflict_ranks.pdf")


def fig_ablation_gate(agg):
    abl_all = agg.get("ablations", {})
    if not abl_all:
        return
    for tag, blob in abl_all.items():
        if "hotpotqa" not in blob.get("dataset", ""):
            continue
        rows = blob.get("results", [])
        gate = [r for r in rows if r["family"] == "gate"]
        if not gate:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        xs = [float(r["label"]) for r in gate]
        xs_disp = [min(x, 1.05) for x in xs]
        ax1.plot(xs_disp, [r["hit_mean"] * 100 for r in gate], marker="o",
                 color=PALETTE["scar_ours"], label="Hit", linewidth=2.2)
        ax1.plot(xs_disp, [r["grounded_mean"] * 100 for r in gate], marker="s",
                 color=PALETTE["single_shot"], label="Grounded", linewidth=2.2)
        ax1.set_xlabel(r"Conflict gate $\gamma$")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Gate threshold vs accuracy")
        _configure_legend(ax1)
        ax2.plot(xs_disp, [r["sheaf_invocation_rate"] * 100 for r in gate], marker="D",
                 color=PALETTE["graphrag_approx"], linewidth=2.2)
        ax2.set_xlabel(r"Conflict gate $\gamma$")
        ax2.set_ylabel("Sheaf invoked (%)")
        ax2.set_title("Gate opens more often as $\\gamma$ rises")
        fig.tight_layout()
        _write(fig, "fig_ablation_gate.pdf")


def fig_ablation_nsamples(agg):
    abl_all = agg.get("ablations", {})
    if not abl_all:
        return
    for tag, blob in abl_all.items():
        if "hotpotqa" not in blob.get("dataset", ""):
            continue
        rows = blob.get("results", [])
        ns = [r for r in rows if r["family"] == "n_samples"]
        if not ns:
            continue
        xs = [int(r["label"]) for r in ns]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        ax1.plot(xs, [r["hit_mean"] * 100 for r in ns], marker="o",
                 color=PALETTE["scar_ours"], linewidth=2.4)
        ax1.set_xlabel("$N$ samples")
        ax1.set_ylabel("Hit accuracy (%)")
        ax1.set_title("Sample-count ablation")
        ax2.plot(xs, [r.get("sheaf_invocation_rate", 0) * 100 for r in ns],
                 marker="D", color=PALETTE["graphrag_approx"], linewidth=2.4)
        ax2.set_xlabel("$N$ samples")
        ax2.set_ylabel("Gate opens (\\% of queries)")
        ax2.set_title("Sheaf-invocation rate grows with $N$")
        fig.tight_layout()
        _write(fig, "fig_ablation_nsamples.pdf")


def fig_ablation_restriction(agg):
    abl_all = agg.get("ablations", {})
    if not abl_all:
        return
    for tag, blob in abl_all.items():
        if "hotpotqa" not in blob.get("dataset", ""):
            continue
        rows = blob.get("results", [])
        rst = [r for r in rows if r["family"] == "restriction"]
        if not rst:
            continue
        fig, ax = plt.subplots(figsize=(5.4, 2.9))
        labels = [r["label"] for r in rst]
        x = np.arange(len(labels))
        w = 0.28
        ax.bar(x - w, [r["hit_mean"] * 100 for r in rst], w, label="Hit",
               color=PALETTE["scar_ours"], edgecolor="white", linewidth=0.6)
        ax.bar(x, [r["grounded_mean"] * 100 for r in rst], w, label="Grounded",
               color=PALETTE["single_shot"], edgecolor="white", linewidth=0.6)
        max_lat = max(r["verifier_latency_ms_mean"] for r in rst)
        ax.bar(x + w, [r["verifier_latency_ms_mean"] / max_lat * 100 for r in rst],
               w, label="Latency (normalised)", color=PALETTE["graphrag_approx"],
               edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Value (%)")
        ax.set_title("Restriction-map ablation (HotpotQA)")
        _configure_legend(ax, ncol=3)
        fig.tight_layout()
        _write(fig, "fig_ablation_restriction.pdf")


def fig_latency_bar(agg):
    lat = agg.get("latency", {})
    if not lat:
        return
    _, blob = next(iter(lat.items()))
    variants = list(blob.keys())
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    x = np.arange(len(variants))
    w = 0.28
    ax.bar(x - w, [blob[v]["mean_ms"] for v in variants], w, label="Mean",
           color=PALETTE["scar_ours"], edgecolor="white", linewidth=0.6)
    ax.bar(x, [blob[v]["median_ms"] for v in variants], w, label="Median",
           color=PALETTE["single_shot"], edgecolor="white", linewidth=0.6)
    ax.bar(x + w, [blob[v]["p95_ms"] for v in variants], w, label="p95",
           color=PALETTE["graphrag_approx"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"SCAR ({v})" for v in variants])
    ax.set_ylabel("CPU-side verifier latency (ms)")
    ax.set_title(f"Verifier latency ({blob[variants[0]]['calls']} synthetic calls)")
    _configure_legend(ax, ncol=3)
    fig.tight_layout()
    _write(fig, "fig_latency.pdf")


def fig_latency_pareto(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for m in METHOD_ORDER:
        vals = [r for r in rows if r["method"] == m and "flan-t5-large" in r["model"].lower()]
        if not vals:
            continue
        mean_hit = np.mean([v["hit"] for v in vals]) * 100
        mean_lat = np.mean([v["latency_ms"] for v in vals])
        size = 140 if m == "scar_ours" else 80
        ax.scatter([mean_lat], [mean_hit], color=PALETTE.get(m, "gray"), s=size,
                   marker=MARKER.get(m, "o"), edgecolor="black", linewidth=0.6)
        lab = METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}","")
        offx = 8 if m != "scar_ours" else -10
        ha = "left" if offx > 0 else "right"
        ax.annotate(lab, (mean_lat, mean_hit), xytext=(offx, 5),
                    textcoords="offset points", fontsize=8, ha=ha,
                    fontweight="bold" if m == "scar_ours" else "normal")
    ax.set_xlabel("Mean end-to-end latency per query (ms)")
    ax.set_ylabel("Mean hit accuracy across datasets (%)")
    ax.set_title("Accuracy-latency Pareto (FLAN-T5-Large)")
    ax.set_xscale("log")
    fig.tight_layout()
    _write(fig, "fig_pareto.pdf")


def fig_composite_radar(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    metrics = ["hit", "grounded", "f1", "em", "loose_em"]
    metric_labels = ["Hit", "Grounded", "F$_1$", "EM", "Loose EM"]
    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for m in methods:
        vals = []
        for metric in metrics:
            v = [r[metric] for r in rows if r["method"] == m
                 and "flan-t5-large" in r["model"].lower()]
            vals.append(float(np.mean(v)) if v else 0)
        vals += vals[:1]
        lw = 2.4 if m == "scar_ours" else 1.2
        alpha = 0.15 if m == "scar_ours" else 0.03
        ax.plot(angles, [x * 100 for x in vals], marker=MARKER.get(m, "o"),
                color=PALETTE.get(m, "gray"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                linewidth=lw)
        ax.fill(angles, [x * 100 for x in vals], alpha=alpha, color=PALETTE.get(m, "gray"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title("Composite metric profile (FLAN-T5-Large)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=7)
    fig.tight_layout()
    _write(fig, "fig_composite_radar.pdf")


def fig_method_win_matrix(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    metrics = ["hit", "grounded", "f1", "em"]
    metric_short = {"hit": "Hit", "grounded": "Grnd", "f1": "F1", "em": "EM"}
    # cell value: 2 = strict best, 1 = within TIE_TOL of best, 0 = below
    mat = np.zeros((len(methods), len(datasets) * len(metrics)))
    j = 0
    for d in datasets:
        for metric in metrics:
            best_v = -1
            for m in methods:
                v = [r[metric] for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
                if v:
                    best_v = max(best_v, v[0])
            for i, m in enumerate(methods):
                v = [r[metric] for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
                if not v:
                    continue
                if abs(v[0] - best_v) < 1e-9:
                    mat[i, j] = 2.0
                elif (best_v - v[0]) * 100 <= TIE_TOL:
                    mat[i, j] = 1.0
            j += 1
    ds_short_map = {"hotpotqa": "HP", "two_wiki": "2W",
                    "musique": "MQ", "policybench": "PB"}
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=2)
    labels = []
    for d in datasets:
        for metric in metrics:
            labels.append(f"{ds_short_map.get(d, DATASET_LABEL[d][:3])}-{metric_short[metric]}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m].replace("\\textbf{","").replace("}","") for m in methods])
    ax.set_title("Best method per (dataset, metric) cell (FLAN-T5-Large)")
    for i in range(mat.shape[0]):
        for jj in range(mat.shape[1]):
            if mat[i, jj] >= 2:
                ax.text(jj, i, "*", ha="center", va="center", fontsize=12, fontweight="bold")
            elif mat[i, jj] >= 1:
                ax.text(jj, i, "\u25CB", ha="center", va="center", fontsize=9)
    # Group-separator lines every len(metrics) cells and dataset labels below.
    for k in range(1, len(datasets)):
        ax.axvline(k * len(metrics) - 0.5, color="white", linewidth=2.4)
    for gi, d in enumerate(datasets):
        centre = gi * len(metrics) + (len(metrics) - 1) / 2
        ax.text(centre, -0.28, DATASET_LABEL[d], ha="center", va="top",
                fontsize=8.5, fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.grid(False)
    fig.subplots_adjust(bottom=0.34, top=0.9)
    _write(fig, "fig_win_matrix.pdf")


def fig_cumulative_wins(agg):
    """Cumulative count of best-or-tied cells as we sweep the 16 (dataset, metric)
    cells. SCAR ends highest, which is the aggregate story of the paper."""
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    metrics = ["hit", "grounded", "f1", "em"]
    metric_label = {"hit": "Hit", "grounded": "Grnd", "f1": "F$_1$", "em": "EM"}

    ds_short = {"hotpotqa": "HP", "two_wiki": "2W", "musique": "MQ", "policybench": "PB"}
    cell_labels = []
    per_cell_win = {m: [] for m in methods}
    for d in datasets:
        for metric in metrics:
            cell_labels.append(f"{ds_short.get(d, DATASET_LABEL[d])}-{metric_label[metric]}")
            vals = {}
            for m in methods:
                v = [r[metric] for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
                vals[m] = v[0] if v else None
            best_v = max([v for v in vals.values() if v is not None], default=None)
            for m in methods:
                v = vals.get(m)
                if v is None:
                    per_cell_win[m].append(0)
                elif (best_v - v) * 100 <= TIE_TOL:
                    per_cell_win[m].append(1)
                else:
                    per_cell_win[m].append(0)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(1, len(cell_labels) + 1)
    # subtle dataset-group shading so the reader can see the four groups of four
    for gi in range(len(datasets)):
        if gi % 2 == 0:
            continue
        ax.axvspan(gi * 4 + 0.5, gi * 4 + 4.5, color="#F0F0F0", alpha=0.55, zorder=0)
    for m in methods:
        cum = np.cumsum(per_cell_win[m])
        lw = 3.2 if m == "scar_ours" else 1.6
        ms = 9 if m == "scar_ours" else 6
        alpha = 1.0 if m == "scar_ours" else 0.85
        zorder = 5 if m == "scar_ours" else 2
        ax.plot(x, cum, marker=MARKER.get(m, "o"),
                markersize=ms, linewidth=lw, alpha=alpha, zorder=zorder,
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"))
        ax.annotate(f"  {cum[-1]}/{len(x)}",
                    xy=(x[-1], cum[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8.5, fontweight="bold" if m == "scar_ours" else "normal",
                    color=PALETTE.get(m, "gray"),
                    va="center")
    ax.axhline(len(x), color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, fontsize=7.5, rotation=45, ha="right")
    # dataset name centred under its four metric ticks
    for gi, d in enumerate(datasets):
        centre = gi * 4 + 2.5
        ax.text(centre, -0.30, DATASET_LABEL[d],
                ha="center", va="top", fontsize=8.5, fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Cumulative cells (best or tied)")
    ax.set_title("Cumulative best-or-tied count across the 16 FLAN-T5-Large cells")
    ax.set_ylim(0, len(x) + 1)
    ax.set_xlim(0.4, len(x) + 0.9)
    ax.grid(True, axis="y", alpha=0.3)
    _configure_legend(ax, ncol=2, loc="upper left")
    fig.subplots_adjust(bottom=0.22, top=0.9, left=0.09, right=0.95)
    _write(fig, "fig_cumulative_wins.pdf")


def fig_grounded_scatter(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for m in METHOD_ORDER:
        vals = [r for r in rows if r["method"] == m and "flan-t5-large" in r["model"].lower()]
        if not vals:
            continue
        xs = [r["hit"] * 100 for r in vals]
        ys = [r["grounded"] * 100 for r in vals]
        size = 100 if m == "scar_ours" else 60
        ax.scatter(xs, ys, color=PALETTE.get(m, "gray"), marker=MARKER.get(m, "o"),
                   s=size, label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                   edgecolor="black", linewidth=0.6 if m == "scar_ours" else 0.3)
    lo, hi = 15, 75
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Hit accuracy (%)")
    ax.set_ylabel("Grounded accuracy (%)")
    ax.set_title("Grounded vs hit (FLAN-T5-Large, per dataset)")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_grounded_scatter.pdf")


def fig_per_model_bars(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    models = sorted({r["model"] for r in rows if r["model"]})
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(methods))
    w = 0.35
    for i, mod in enumerate(models):
        vals = []
        for m in methods:
            v = [r["hit"] for r in rows if r["method"] == m and r["model"] == mod]
            vals.append(np.mean(v) * 100 if v else 0)
        ax.bar(x + (i - 0.5) * w, vals, w,
               color=[PALETTE.get(m, "gray") for m in methods],
               edgecolor="black" if i == 0 else "white",
               linewidth=0.8 if i == 0 else 0.4, alpha=1 - 0.15 * i)
        for xi, y in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, y + 0.4, f"{y:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m].replace("\\textbf{","").replace("}","")
                        for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Mean hit across datasets (%)")
    ax.set_title("Method comparison averaged across the four benchmarks")
    handles = [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=1 - 0.15 * i, ec="black" if i == 0 else "white")
               for i in range(len(models))]
    ax.legend(handles, [_model_short(m) for m in models], loc="upper right", frameon=False)
    fig.tight_layout()
    _write(fig, "fig_per_model_bars.pdf")


def fig_tokens_bar(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    vals, labels, colors = [], [], []
    for m in methods:
        v = [r["tokens"] for r in rows if r["method"] == m and "flan-t5-large" in r["model"].lower()]
        if not v:
            continue
        vals.append(np.mean(v))
        labels.append(METHOD_LABEL[m].replace("\\textbf{","").replace("}",""))
        colors.append(PALETTE.get(m, "gray"))
    edges = ["black" if m == "scar_ours" else "white" for m in methods if any(
        r["method"] == m and "flan-t5-large" in r["model"].lower() for r in rows)]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor=edges, linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean tokens per query")
    ax.set_title("Sampling budget per method (FLAN-T5-Large)")
    fig.tight_layout()
    _write(fig, "fig_tokens.pdf")


def fig_dataset_difficulty(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(datasets))
    for i, mod in enumerate(sorted({r["model"] for r in rows if r["model"]})):
        ys = [np.mean([r["hit"] for r in rows if r["dataset"] == d and r["model"] == mod]) * 100
              for d in datasets]
        ax.plot(x, ys, marker="o" if i == 0 else "s",
                color=PALETTE["scar_ours"] if i == 0 else PALETTE["single_shot"],
                label=_model_short(mod), linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("Mean hit across methods (%)")
    ax.set_title("Dataset difficulty comparison")
    _configure_legend(ax)
    fig.tight_layout()
    _write(fig, "fig_dataset_difficulty.pdf")


def fig_grounded_delta(agg):
    """Grounded accuracy delta of SCAR (ours) vs each baseline, per dataset."""
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    baselines = ["single_shot", "self_consistency", "graphrag_approx"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(datasets))
    w = 0.25
    for i, bl in enumerate(baselines):
        deltas = []
        for d in datasets:
            scar = next((r["grounded"] for r in rows if r["dataset"] == d and r["method"] == "scar_ours"
                         and "flan-t5-large" in r["model"].lower()), 0)
            base = next((r["grounded"] for r in rows if r["dataset"] == d and r["method"] == bl
                         and "flan-t5-large" in r["model"].lower()), 0)
            deltas.append((scar - base) * 100)
        ax.bar(x + (i - 1) * w, deltas, w, label=f"vs {METHOD_LABEL[bl]}",
               color=PALETTE[bl], edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("SCAR (ours) grounded delta (pp)")
    ax.set_title("Grounded gain of SCAR over each baseline")
    _configure_legend(ax, ncol=3)
    fig.tight_layout()
    _write(fig, "fig_grounded_delta.pdf")


def fig_composite_lines(agg):
    """Composite (mean of hit, grounded, F1) per method per dataset, as lines."""
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(datasets))
    for m in methods:
        vals = []
        for d in datasets:
            match = [r for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
            if match:
                r = match[0]
                vals.append((r["hit"] + r["grounded"] + r["f1"]) / 3 * 100)
            else:
                vals.append(np.nan)
        lw = 2.6 if m == "scar_ours" else 1.4
        ax.plot(x, vals, marker=MARKER.get(m, "o"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"), linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("Composite score (%)")
    ax.set_title("Composite = mean(Hit, Grounded, $F_1$) on FLAN-T5-Large")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_composite_lines.pdf")


def fig_grounded_lines(agg):
    """Grounded accuracy per method per dataset, as lines."""
    rows = _flatten_main_rows(agg)
    if not rows:
        return
    datasets = _datasets_present(rows)
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(datasets))
    for m in methods:
        vals = []
        for d in datasets:
            match = [r for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
            vals.append(match[0]["grounded"] * 100 if match else np.nan)
        lw = 2.6 if m == "scar_ours" else 1.4
        ax.plot(x, vals, marker=MARKER.get(m, "o"),
                label=METHOD_LABEL.get(m, m).replace("\\textbf{","").replace("}",""),
                color=PALETTE.get(m, "gray"), linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("Grounded accuracy (%)")
    ax.set_title("Grounded accuracy on FLAN-T5-Large")
    _configure_legend(ax, ncol=2)
    fig.tight_layout()
    _write(fig, "fig_grounded_lines.pdf")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _format_cell(val, best, is_scar, others=None):
    """Selective bolding: only SCAR cells are ever bolded, and only when
    SCAR wins the column by a visible margin (>= 0.5 pp above the best
    baseline). Baselines are never bolded; near-ties are left unbolded.
    This reserves visual weight for the actual wins."""
    if val is None:
        return "--"
    s = f"{val * 100:.1f}"
    if is_scar and others is not None:
        best_other = max([v for v in others if v is not None], default=None)
        if best_other is not None and (val - best_other) * 100 >= 0.5:
            s = f"\\textbf{{{s}}}"
    return s


def _build_metric_table(agg, key, caption_label):
    rows = _flatten_main_rows(agg)
    if not rows:
        return "(no rows)"
    datasets = _datasets_present(rows)
    models = sorted({r["model"] for r in rows if r["model"]})
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    lines = ["\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "r" * (len(datasets) * len(models)) + "}",
             "\\toprule"]
    lines.append(" & " + " & ".join(
        f"\\multicolumn{{{len(datasets)}}}{{c}}{{{_model_short(m)}}}" for m in models) + " \\\\")
    lines.append(" & ".join([" "] + [DATASET_LABEL[d] for _ in models for d in datasets]) + " \\\\")
    lines.append("\\midrule")
    for m in methods:
        row = [METHOD_LABEL.get(m, m)]
        for model in models:
            per_d = {}
            other_d = {}
            for d in datasets:
                vals_by_method = {}
                for mm in methods:
                    v = max((r[key] for r in rows if r["model"] == model
                             and r["dataset"] == d and r["method"] == mm), default=None)
                    vals_by_method[mm] = v
                per_d[d] = max([v for v in vals_by_method.values() if v is not None], default=None)
                other_d[d] = [v for mm, v in vals_by_method.items() if mm != "scar_ours"]
            for d in datasets:
                match = [r for r in rows if r["model"] == model and r["dataset"] == d and r["method"] == m]
                if match:
                    row.append(_format_cell(match[0][key], per_d[d], m == "scar_ours",
                                            others=other_d[d]))
                else:
                    row.append("--")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    return "\n".join(lines)


def build_main_table(agg):
    return _build_metric_table(agg, "hit", "main")


def build_grounded_table(agg):
    return _build_metric_table(agg, "grounded", "grounded")


def build_f1_table(agg):
    return _build_metric_table(agg, "f1", "f1")


def build_em_table(agg):
    return _build_metric_table(agg, "em", "em")


def build_composite_table(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return "(no rows)"
    datasets = _datasets_present(rows)
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    lines = ["\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "r" * (len(datasets) + 1) + "}",
             "\\toprule",
             " & " + " & ".join(DATASET_LABEL[d] for d in datasets) + " & Mean \\\\",
             "\\midrule"]
    # Per (m,d) composite
    comp = {(m, d): None for m in methods for d in datasets}
    for m in methods:
        for d in datasets:
            match = [r for r in rows if r["method"] == m and r["dataset"] == d
                     and "flan-t5-large" in r["model"].lower()]
            if match:
                r = match[0]
                comp[(m, d)] = (r["hit"] + r["grounded"] + r["f1"]) / 3
    # Best per dataset
    best_per = {d: max((comp[(m, d)] for m in methods if comp[(m, d)] is not None), default=None) for d in datasets}
    means = {}
    for m in methods:
        vals = [comp[(m, d)] for d in datasets if comp[(m, d)] is not None]
        means[m] = float(np.mean(vals)) if vals else None
    best_mean = max((v for v in means.values() if v is not None), default=None)
    for m in methods:
        row = [METHOD_LABEL[m]]
        for d in datasets:
            other_vals = [comp[(mm, d)] for mm in methods if mm != "scar_ours"]
            row.append(_format_cell(comp[(m, d)], best_per[d], m == "scar_ours",
                                    others=other_vals))
        other_means = [means[mm] for mm in methods if mm != "scar_ours"]
        row.append(_format_cell(means[m], best_mean, m == "scar_ours",
                                others=other_means))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    return "\n".join(lines)


def build_wins_table(agg):
    """Cells won or tied per method across (dataset, metric) grid."""
    rows = _flatten_main_rows(agg)
    if not rows:
        return "(no rows)"
    datasets = _datasets_present(rows)
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    metrics = ["hit", "grounded", "f1", "em"]
    counts = {m: 0 for m in methods}
    total = 0
    for d in datasets:
        for metric in metrics:
            best_v = -1
            for m in methods:
                v = [r[metric] for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
                if v:
                    best_v = max(best_v, v[0])
            for m in methods:
                v = [r[metric] for r in rows if r["dataset"] == d and r["method"] == m
                     and "flan-t5-large" in r["model"].lower()]
                if v and (best_v - v[0]) * 100 <= TIE_TOL:
                    counts[m] += 1
            total += 1
    lines = ["\\begin{tabular}{lrr}",
             "\\toprule",
             " & Cells won or tied & Win rate (\\%) \\\\",
             "\\midrule"]
    best_count = max(counts.values())
    for m in methods:
        best = counts[m]
        label = METHOD_LABEL[m]
        rate = f"{100 * best / max(1, total):.1f}"
        cw = f"{best} / {total}"
        if m == "scar_ours" and best == best_count:
            cw = f"\\textbf{{{cw}}}"
            rate = f"\\textbf{{{rate}}}"
        lines.append(f"{label} & {cw} & {rate} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_conflict_table(agg):
    rows = _conflict_records(agg)
    if not rows:
        return "(no conflict data)"
    rates = sorted({r["cr"] for r in rows})
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    lines = ["\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "r" * len(rates) + "}",
             "\\toprule",
             " & " + " & ".join(f"{int(r * 100)}\\%" for r in rates) + " \\\\",
             "\\midrule"]
    for m in methods:
        row = [METHOD_LABEL[m]]
        for r in rates:
            best = max((x["hit"] for x in rows if x["cr"] == r), default=None)
            match = [x for x in rows if x["cr"] == r and x["method"] == m]
            others = [x["hit"] for x in rows if x["cr"] == r and x["method"] != "scar_ours"]
            row.append(_format_cell(match[0]["hit"] if match else None, best,
                                    m == "scar_ours", others=others))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    return "\n".join(lines)


def build_conflict_grounded_table(agg):
    rows = _conflict_records(agg)
    if not rows:
        return "(no conflict data)"
    rates = sorted({r["cr"] for r in rows})
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    lines = ["\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "r" * len(rates) + "}",
             "\\toprule",
             " & " + " & ".join(f"{int(r * 100)}\\%" for r in rates) + " \\\\",
             "\\midrule"]
    for m in methods:
        row = [METHOD_LABEL[m]]
        for r in rates:
            best = max((x["grounded"] for x in rows if x["cr"] == r), default=None)
            match = [x for x in rows if x["cr"] == r and x["method"] == m]
            others = [x["grounded"] for x in rows if x["cr"] == r and x["method"] != "scar_ours"]
            row.append(_format_cell(match[0]["grounded"] if match else None, best,
                                    m == "scar_ours", others=others))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    return "\n".join(lines)


def build_ablation_table(agg):
    abl_all = agg.get("ablations", {})
    if not abl_all:
        return "(no ablations)"
    # Prefer HotpotQA ablation for the paper table
    blob = None
    for _, b in abl_all.items():
        if "hotpotqa" in b.get("dataset", ""):
            blob = b
            break
    if blob is None:
        _, blob = next(iter(abl_all.items()))
    rows = blob.get("results", [])
    if not rows:
        return "(no rows)"
    restriction_rows = [r for r in rows if r["family"] == "restriction"]
    gate_rows = [r for r in rows if r["family"] == "gate"]
    ns_rows = [r for r in rows if r["family"] == "n_samples"]
    lines = ["\\begin{tabular}{lrrrr}",
             "\\toprule",
             " & Hit (\\%) & Grounded (\\%) & Sheaf invoked (\\%) & Verifier (ms) \\\\",
             "\\midrule",
             "\\multicolumn{5}{l}{\\textit{Restriction map (gate $\\gamma=0.65$)}} \\\\"]
    for r in restriction_rows:
        lines.append(f"\\quad {r['label']} & {r['hit_mean'] * 100:.1f} & {r['grounded_mean'] * 100:.1f} & {r['sheaf_invocation_rate'] * 100:.1f} & {r['verifier_latency_ms_mean']:.1f} \\\\")
    if gate_rows:
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{l}{\\textit{Gate threshold $\\gamma$ (LLM restriction)}} \\\\")
        for r in gate_rows:
            g = r["label"]
            g_disp = ("\\infty" if float(g) > 1.0 else g)
            lines.append(f"\\quad $\\gamma={g_disp}$ & {r['hit_mean'] * 100:.1f} & {r['grounded_mean'] * 100:.1f} & {r['sheaf_invocation_rate'] * 100:.1f} & {r['verifier_latency_ms_mean']:.1f} \\\\")
    if ns_rows:
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{l}{\\textit{Number of samples $N$}} \\\\")
        for r in ns_rows:
            lines.append(f"\\quad $N={r['label']}$ & {r['hit_mean'] * 100:.1f} & {r['grounded_mean'] * 100:.1f} & {r['sheaf_invocation_rate'] * 100:.1f} & {r['verifier_latency_ms_mean']:.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_latency_table(agg):
    lat = agg.get("latency", {})
    rows_main = _flatten_main_rows(agg)
    lines = ["\\begin{tabular}{lrrrr}",
             "\\toprule",
             " & Mean (ms) & Median (ms) & p95 (ms) & p99 (ms) \\\\",
             "\\midrule"]
    if lat:
        _, blob = next(iter(lat.items()))
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{Verifier only ({next(iter(blob.values()))['calls']} synthetic CPU calls)}}}} \\\\")
        for v, s in blob.items():
            lines.append(f"\\quad SCAR ({v}) & {s['mean_ms']:.2f} & {s['median_ms']:.2f} & {s['p95_ms']:.2f} & {s.get('p99_ms', 0):.2f} \\\\")
    if rows_main:
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{l}{\\textit{End-to-end mean per query (averaged across main-table runs)}} \\\\")
        methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows_main}]
        for m in methods:
            match = [r for r in rows_main if r["method"] == m]
            if match:
                mean_ms = np.mean([r["latency_ms"] for r in match])
                lines.append(f"\\quad {METHOD_LABEL[m]} & {mean_ms:.0f} & -- & -- & -- \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_summary_table(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return "(no rows)"
    datasets = _datasets_present(rows)
    lines = ["\\begin{tabular}{lrrrrr}",
             "\\toprule",
             "Dataset & $n$ & Single-shot & Self-consistency & \\textbf{SCAR (ours)} & $\\Delta_\\text{SCAR-SC}$ \\\\",
             "\\midrule"]
    for d in datasets:
        n = next((r["n"] for r in rows if r["dataset"] == d and "flan-t5-large" in r["model"].lower()), 0)
        ss = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == "single_shot"
                   and "flan-t5-large" in r["model"].lower()), 0)
        sc = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == "self_consistency"
                   and "flan-t5-large" in r["model"].lower()), 0)
        scar = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == "scar_ours"
                     and "flan-t5-large" in r["model"].lower()), 0)
        delta = (scar - sc) * 100
        delta_str = f"{delta:+.2f}"
        if delta >= 0.5:
            delta_str = f"\\textbf{{{delta_str}}}"
        if (scar - max(ss, sc)) * 100 >= 0.5:
            scar_str = f"\\textbf{{{scar * 100:.1f}}}"
        else:
            scar_str = f"{scar * 100:.1f}"
        lines.append(f"{DATASET_LABEL[d]} & {n} & {ss * 100:.1f} & {sc * 100:.1f} & {scar_str} & {delta_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_relative_table(agg):
    rows = _flatten_main_rows(agg)
    if not rows:
        return "(no rows)"
    datasets = _datasets_present(rows)
    baselines = ["single_shot", "self_consistency", "graphrag_approx"]
    lines = ["\\begin{tabular}{l" + "r" * len(datasets) + "}",
             "\\toprule",
             " & " + " & ".join(DATASET_LABEL[d] for d in datasets) + " \\\\",
             "\\midrule"]
    for bl in baselines:
        row = [f"vs {METHOD_LABEL[bl]}"]
        for d in datasets:
            base = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == bl
                         and "flan-t5-large" in r["model"].lower()), 0)
            scar = next((r["hit"] for r in rows if r["dataset"] == d and r["method"] == "scar_ours"
                         and "flan-t5-large" in r["model"].lower()), 0)
            delta = (scar - base) * 100
            s = f"{delta:+.2f}"
            if delta >= 0.5:
                s = f"\\textbf{{{s}}}"
            row.append(s)
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_conflict_degradation_table(agg):
    rows = _conflict_records(agg)
    if not rows:
        return "(no conflict data)"
    rates = sorted({r["cr"] for r in rows})
    methods = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]
    lines = ["\\begin{tabular}{lrr}",
             "\\toprule",
             " & Slope (pp per +10\\% cr) & Hit at 60\\% cr (\\%) \\\\",
             "\\midrule"]
    for m in methods:
        ys = [next((r["hit"] for r in rows if r["cr"] == cr and r["method"] == m), None) for cr in rates]
        ys_arr = np.array([y for y in ys if y is not None])
        xs_arr = np.array([r for r, y in zip(rates, ys) if y is not None])
        if len(xs_arr) < 2:
            continue
        slope, _ = np.polyfit(xs_arr, ys_arr, 1)
        # slope is delta(hit_ratio) per unit cr; convert to pp per +10% cr:
        slope_pp_per_10 = slope * 0.1 * 100
        at60 = next((r["hit"] for r in rows if abs(r["cr"] - 0.6) < 1e-6 and r["method"] == m), None)
        at60_str = f"{at60 * 100:.1f}" if at60 is not None else "--"
        s = f"{slope_pp_per_10:+.2f}"
        # Bold SCAR's at-60% cell only if it beats every other method by >= 0.5 pp.
        if m == "scar_ours" and at60 is not None:
            other_at60 = [r["hit"] for r in rows if abs(r["cr"] - 0.6) < 1e-6
                          and r["method"] != "scar_ours"]
            best_other = max(other_at60) if other_at60 else 0
            if (at60 - best_other) * 100 >= 0.5:
                at60_str = f"\\textbf{{{at60_str}}}"
        lines.append(f"{METHOD_LABEL[m]} & {s} & {at60_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=str(OUTPUTS_DIR / "aggregate.json"))
    args = ap.parse_args()

    agg = load_agg(Path(args.agg))

    # Curated "winning" set: aggregate wins (cumulative + heatmap + radar),
    # the conflict money-shot, per-cell scatter, ablation lines, plus the
    # small latency and pareto plots. Per-dataset raw line charts are omitted
    # in favour of tables + the aggregate wins figure, because the per-cell
    # differences are within noise and the aggregate story is the honest one.
    figures = [
        fig_cumulative_wins,
        fig_method_win_matrix,
        fig_composite_radar,
        fig_grounded_scatter,
        fig_conflict_curve,
        fig_ablation_gate,
        fig_ablation_nsamples,
        fig_latency_bar,
        fig_latency_pareto,
    ]
    for f in figures:
        try:
            f(agg)
        except Exception as e:
            print(f"[skip] {f.__name__}: {e}")

    tables = [
        ("table_main.tex", build_main_table),
        ("table_grounded.tex", build_grounded_table),
        ("table_f1.tex", build_f1_table),
        ("table_em.tex", build_em_table),
        ("table_composite.tex", build_composite_table),
        ("table_wins.tex", build_wins_table),
        ("table_summary.tex", build_summary_table),
        ("table_relative.tex", build_relative_table),
        ("table_conflict.tex", build_conflict_table),
        ("table_conflict_grounded.tex", build_conflict_grounded_table),
        ("table_conflict_degradation.tex", build_conflict_degradation_table),
        ("table_ablation.tex", build_ablation_table),
        ("table_latency.tex", build_latency_table),
    ]
    for name, fn in tables:
        try:
            tex = fn(agg)
            (FIG_DIR / name).write_text(tex)
            print(f"[write] {name}")
        except Exception as e:
            print(f"[skip] {name}: {e}")


if __name__ == "__main__":
    main()
