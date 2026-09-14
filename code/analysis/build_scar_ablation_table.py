#!/usr/bin/env python3
"""Consolidate the four SCAR variants (default/tuned x extras/no-extras)
across all 8 (dataset, model) corpora into a single LaTeX table for the
manuscript, and also compute the per-baseline delta so a reader can see
exactly what per-corpus tuning and the extended candidate pool each buy.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "outputs" / "ablations" / "scar_variants"
FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


DATASET_LABEL = {"hotpotqa": "HotpotQA", "two_wiki": "2WikiMH",
                 "twowiki": "2WikiMH",
                 "musique": "MuSiQue", "policybench": "PolicyBench"}


def _lookup(stem: str) -> tuple[str, str]:
    ds = "?"
    for k, v in DATASET_LABEL.items():
        if k in stem:
            ds = v
            break
    model = "FLAN-T5-Large" if "flant5large" in stem else "Qwen 2.5-0.5B"
    return model, ds


def main():
    rows = []
    for p in sorted(IN_DIR.glob("*_ablation.json")):
        d = json.load(open(p))
        model, ds = _lookup(p.stem)
        row = {"model": model, "dataset": ds, "stem": p.stem}
        for k, v in d["variants"].items():
            row[k] = v["summ"]["hit"] * 100
        # baselines: read main summary to get single-shot / self-consistency
        base_path = ROOT / "outputs" / "main" / f"{p.stem.replace('_ablation','')}_summary.json"
        base_path = ROOT / "outputs" / "main" / f"{p.stem.replace('_ablation','')}.json"
        # Correct: summary file name pattern
        base_stem = p.stem.replace("_ablation", "")
        base_path = ROOT / "outputs" / "main" / f"{base_stem}_summary.json"
        if base_path.exists():
            b = json.load(open(base_path))
            m = b["results"]["methods"]
            row["single_shot"] = m.get("single_shot", {}).get("hit_mean", 0) * 100
            row["self_consistency"] = m.get("self_consistency", {}).get("hit_mean", 0) * 100
            row["graphrag_approx"] = m.get("graphrag_approx", {}).get("hit_mean", 0) * 100
        rows.append(row)

    # Save the raw consolidated JSON
    (IN_DIR / "consolidated.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(f"[write] {IN_DIR/'consolidated.json'}")

    # Pretty print
    print()
    hdr = ("model", "dataset", "SS", "SC", "GR",
           "SCAR-D-NE", "SCAR-D-E", "SCAR-T-NE", "SCAR-T-E")
    print(("{:14s}  {:10s}  " + "  ".join(["{:>6s}"] * 7)).format(*hdr))
    for r in rows:
        vals = (r.get("single_shot", 0), r.get("self_consistency", 0),
                r.get("graphrag_approx", 0),
                r["default_noextras"], r["default_extras"],
                r["tuned_noextras"], r["tuned_extras"])
        print(("{:14s}  {:10s}  " + "  ".join(["{:6.2f}"] * 7)).format(
            r["model"], r["dataset"], *vals))

    # LaTeX table: rows are (model, dataset), columns are baselines + SCAR variants
    ordered_ds = ["HotpotQA", "2WikiMH", "MuSiQue", "PolicyBench"]
    lines = ["\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{llrrrrrrr}", "\\toprule",
             " & & \\multicolumn{3}{c}{Baselines} & "
             "\\multicolumn{4}{c}{SCAR variants (hit, \\%)} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}",
             "Backbone & Dataset & SS & SC & GR & "
             "default & default & tuned & tuned \\\\",
             " & & & & & (no extras) & (extras) & (no extras) & (extras) \\\\",
             "\\midrule"]
    for model in ("FLAN-T5-Large", "Qwen 2.5-0.5B"):
        for ds in ordered_ds:
            r = next((x for x in rows if x["model"] == model and x["dataset"] == ds), None)
            if not r:
                continue
            vals = {
                "SS": r.get("single_shot", 0),
                "SC": r.get("self_consistency", 0),
                "GR": r.get("graphrag_approx", 0),
                "SCAR-D-NE": r["default_noextras"],
                "SCAR-D-E": r["default_extras"],
                "SCAR-T-NE": r["tuned_noextras"],
                "SCAR-T-E": r["tuned_extras"],
            }
            best = max(vals.values())
            # Bold the tuned-extras SCAR cell only when it beats every other
            # column in the row by >= 0.5 pp. All other cells render plain.
            others = [v for k, v in vals.items() if k != "SCAR-T-E"]
            best_other = max(others) if others else 0
            def fmt(v, is_headline=False):
                if is_headline and (v - best_other) >= 0.5:
                    return f"\\textbf{{{v:.1f}}}"
                return f"{v:.1f}"
            lines.append(
                f"{model} & {ds} & "
                f"{fmt(vals['SS'])} & "
                f"{fmt(vals['SC'])} & "
                f"{fmt(vals['GR'])} & "
                f"{fmt(vals['SCAR-D-NE'])} & "
                f"{fmt(vals['SCAR-D-E'])} & "
                f"{fmt(vals['SCAR-T-NE'])} & "
                f"{fmt(vals['SCAR-T-E'], is_headline=True)} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}%", "}"]
    tex_path = FIG_DIR / "table_scar_variants.tex"
    tex_path.write_text("\n".join(lines))
    print(f"[write] {tex_path}")

    # Compute average deltas across corpora
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    d_all = [(r["tuned_extras"] - r["default_extras"]) for r in rows]
    e_all = [(r["default_extras"] - r["default_noextras"]) for r in rows]
    d_and_e = [(r["tuned_extras"] - r["default_noextras"]) for r in rows]

    ss_gap = [(r["tuned_extras"] - r.get("single_shot", 0)) for r in rows]
    sc_gap = [(r["tuned_extras"] - r.get("self_consistency", 0)) for r in rows]

    print(f"\n=== Mean deltas across 8 corpora (hit accuracy, percentage points)")
    print(f"  tuning at fixed extras: {mean(d_all):+.2f} pp "
          f"(min {min(d_all):+.2f}, max {max(d_all):+.2f})")
    print(f"  extras at fixed default:{mean(e_all):+.2f} pp "
          f"(min {min(e_all):+.2f}, max {max(e_all):+.2f})")
    print(f"  tuning + extras vs default_noextras: {mean(d_and_e):+.2f} pp "
          f"(min {min(d_and_e):+.2f}, max {max(d_and_e):+.2f})")
    print(f"  SCAR-tuned-extras vs single-shot:   {mean(ss_gap):+.2f} pp")
    print(f"  SCAR-tuned-extras vs self-consist.: {mean(sc_gap):+.2f} pp")


if __name__ == "__main__":
    main()
