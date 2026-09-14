#!/usr/bin/env python3
"""Aggregate all summary JSONs into single tables/figures inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import OUTPUTS_DIR, dump_json


def load_summaries(directory: Path) -> dict:
    out = {}
    for f in sorted(directory.glob("*_summary.json")):
        try:
            out[f.stem.replace("_summary", "")] = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "aggregate.json"))
    args = ap.parse_args()

    all_summaries = {}
    for sub in ("main", "conflict"):
        d = OUTPUTS_DIR / sub
        all_summaries[sub] = load_summaries(d)

    # Ablations file(s)
    abl_dir = OUTPUTS_DIR / "ablations"
    abl = {}
    for f in sorted(abl_dir.glob("*.json")):
        try:
            abl[f.stem] = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f}: {e}")
    all_summaries["ablations"] = abl

    lat_dir = OUTPUTS_DIR / "latency"
    lat = {}
    for f in sorted(lat_dir.glob("*.json")):
        try:
            lat[f.stem] = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f}: {e}")
    all_summaries["latency"] = lat

    dump_json(Path(args.out), all_summaries)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
