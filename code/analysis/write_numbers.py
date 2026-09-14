#!/usr/bin/env python3
"""Emit LaTeX \\newcommand macros with numeric results from the aggregate JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import OUTPUTS_DIR


def fmt_pct(x: float, decimals: int = 1) -> str:
    return f"{100 * x:.{decimals}f}"


def emit_stub() -> str:
    return (
        "% aggregate.json missing, using TBD placeholders\n"
        "\\providecommand{\\numN}{5}\n"
        "\\providecommand{\\numDatasets}{four}\n"
        "\\providecommand{\\numLat}[1]{\\texttt{TBD}}\n"
        "\\providecommand{\\numGain}[1]{\\texttt{TBD}}\n"
        "\\providecommand{\\numConflict}[2]{\\texttt{TBD}}\n"
        "\\providecommand{\\numGate}[1]{0.65}\n"
        "\\providecommand{\\numMain}[2]{\\texttt{TBD}}\n"
        "\\providecommand{\\numFT}[1]{\\texttt{TBD}}\n"
        "\\providecommand{\\numQwen}[1]{\\texttt{TBD}}\n"
    )


def main():
    agg_path = OUTPUTS_DIR / "aggregate.json"
    if not agg_path.exists():
        print(emit_stub())
        return

    agg = json.loads(agg_path.read_text())
    lines: list[str] = []

    n_datasets = len({v.get("dataset") for v in agg.get("main", {}).values()})
    dtext = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n_datasets, str(n_datasets))
    lines.append("\\providecommand{\\numN}{5}")
    lines.append(f"\\providecommand{{\\numDatasets}}{{{dtext}}}")

    # Per-cell hit accuracy, keyed by (dataset, method).
    # Emit as \providecommand{\numMainDATASETMETHOD} then \providecommand{\numMain}[2]{\csname numMain#1#2\endcsname}
    for tag, meta in agg.get("main", {}).items():
        if "results" not in meta:
            continue
        ds = meta["dataset"].replace("-", "").replace("_", "")
        for m, v in meta["results"]["methods"].items():
            key = f"{ds}{m.replace('_','')}"
            lines.append(f"\\expandafter\\providecommand\\csname numMain{key}\\endcsname{{{fmt_pct(v['hit_mean'])}}}")
    lines.append("\\providecommand{\\numMain}[2]{\\csname numMain#1#2\\endcsname}")

    # Primary model = flan-t5-large
    primary_model = None
    for meta in agg.get("main", {}).values():
        m = meta.get("model") or ""
        if "flan-t5-large" in m.lower():
            primary_model = meta["model"]
            break

    # Average gain across datasets for SCAR-LLM vs self-consistency.
    gains = []
    for meta in agg.get("main", {}).values():
        if meta.get("model") != primary_model or "results" not in meta:
            continue
        methods = meta["results"]["methods"]
        sc = methods.get("self_consistency", {}).get("hit_mean")
        sc_llm = methods.get("scar_llm", {}).get("hit_mean")
        if sc is None or sc_llm is None:
            continue
        gains.append((sc_llm - sc) * 100)
    gain_avg = f"{(sum(gains)/len(gains)) if gains else 0:.1f}"
    lines.append(f"\\providecommand{{\\numGainval}}{{{gain_avg}}}")
    lines.append("\\providecommand{\\numGain}[1]{\\numGainval}")

    # Latency: mean across variants and total calls.
    lat = agg.get("latency", {})
    if lat:
        first = next(iter(lat.values()))
        # Report identity restriction map for headline latency numbers.
        identity = first.get("identity", next(iter(first.values())))
        mean_ms = identity.get("mean_ms", 0)
        p95_ms = identity.get("p95_ms", 0)
        calls = identity.get("calls", 0)
        lines.append(f"\\providecommand{{\\numLatmean}}{{{mean_ms:.2f}}}")
        lines.append(f"\\providecommand{{\\numLatpninetyfive}}{{{p95_ms:.2f}}}")
        lines.append(f"\\providecommand{{\\numLatcalls}}{{{calls}}}")
        for variant, stats in first.items():
            safe = variant.replace("-", "").replace("_", "")
            lines.append(f"\\providecommand{{\\numLat{safe}mean}}{{{stats['mean_ms']:.2f}}}")
            lines.append(f"\\providecommand{{\\numLat{safe}pninetyfive}}{{{stats['p95_ms']:.2f}}}")
    else:
        lines.append("\\providecommand{\\numLatmean}{TBD}")
        lines.append("\\providecommand{\\numLatpninetyfive}{TBD}")
        lines.append("\\providecommand{\\numLatcalls}{TBD}")
    lines.append("\\providecommand{\\numLat}[1]{\\csname numLat#1\\endcsname}")

    # Conflict at ~60% rate for headline number
    conflict = agg.get("conflict", {})
    best = None
    best_dist = 999.0
    for v in conflict.values():
        cr = float(v.get("conflict_rate", 0))
        if abs(cr - 0.6) < best_dist:
            best_dist = abs(cr - 0.6)
            best = v
    if best is not None:
        sc_hit = best["results"]["methods"].get("self_consistency", {}).get("hit_mean", 0)
        scar_hit = best["results"]["methods"].get("scar_llm", {}).get("hit_mean", 0)
        lines.append(f"\\providecommand{{\\numConflictsc}}{{{100 * sc_hit:.1f}}}")
        lines.append(f"\\providecommand{{\\numConflictscar}}{{{100 * scar_hit:.1f}}}")
    else:
        lines.append("\\providecommand{\\numConflictsc}{TBD}")
        lines.append("\\providecommand{\\numConflictscar}{TBD}")
    lines.append("\\providecommand{\\numConflict}[1]{\\csname numConflict#1\\endcsname}")

    lines.append("\\providecommand{\\numGate}[1]{0.65}")
    lines.append("\\providecommand{\\numFT}[1]{FLAN-T5-Large}")
    lines.append("\\providecommand{\\numQwen}[1]{Qwen 2.5-0.5B}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
