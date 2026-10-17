# -*- coding: utf-8 -*-
"""
Convert a false-positive rate into alarms per second, from the capture's own
timestamps.

A percentage is the wrong unit for an operator. 2.26% of frames means nothing
until you know how many frames a second the bus carries; on a busy powertrain
bus it can be tens of alarms per second, which is not a workable alarm. This
reads the frame rate out of the capture and does the arithmetic.

    python bus_rate.py
    python bus_rate.py --fpr 0.0226 --fpr 0.0100 --fpr 0.0031

With no --fpr given it reads the rates out of results/zero_day.json, so the
numbers match the ones already reported.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

from preprocessing.loader import load_all                   # noqa: E402


def rate_of(df, source: str) -> Dict[str, float]:
    """Frames per second over one capture, from first to last timestamp.

    Reported with the median inter-arrival gap beside it, because a capture
    with a long idle stretch has a mean rate that describes no part of it.
    """
    sub = df[df["source"] == source]
    t = np.sort(sub["timestamp"].to_numpy(dtype=np.float64))
    if len(t) < 2:
        return {}
    span = float(t[-1] - t[0])
    gaps = np.diff(t)
    gaps = gaps[gaps > 0]
    return {
        "frames": int(len(t)),
        "span_seconds": span,
        "mean_frames_per_second": len(t) / span if span > 0 else float("nan"),
        "median_gap_ms": float(np.median(gaps) * 1000) if len(gaps) else float("nan"),
        "median_frames_per_second": float(1.0 / np.median(gaps)) if len(gaps) else float("nan"),
        "p95_frames_per_second": float(1.0 / np.quantile(gaps, 0.05)) if len(gaps) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpr", type=float, action="append", default=None,
                    help="false-positive rate as a fraction; repeatable")
    ap.add_argument("--out", default=os.path.join(RESULTS, "bus_rate.json"))
    args = ap.parse_args()

    print("Reading the capture to measure its frame rate ...", flush=True)
    df = load_all()
    rates = {s: rate_of(df, s) for s in df["source"].unique()}

    print("\n  Frame rate per capture")
    for src, r in rates.items():
        if not r:
            continue
        print(f"    {src:<10} {r['frames']:>9,} frames over {r['span_seconds']:>9.1f} s")
        print(f"    {'':<10} mean {r['mean_frames_per_second']:>8.1f} f/s   "
              f"median {r['median_frames_per_second']:>8.1f} f/s   "
              f"busiest 5% {r['p95_frames_per_second']:>8.1f} f/s")

    # which rates to convert
    pairs: List[tuple] = []
    if args.fpr:
        pairs = [(f"{f * 100:.2f}% FPR", f) for f in args.fpr]
    else:
        zd_file = os.path.join(RESULTS, "zero_day.json")
        if os.path.exists(zd_file):
            zd = json.load(open(zd_file, encoding="utf-8"))
            pairs = [(n, m["false_positive_rate"]) for n, m in zd.get("models", {}).items()]
        if not pairs:
            pairs = [("illustrative 2.26%", 0.0226), ("illustrative 1.00%", 0.0100)]

    out = {"captures": rates, "alarms_per_second": {}}
    print("\n  False alarms per second, at each capture's mean frame rate")
    head = "    {:<28}{:>12}".format("Detector", "FPR")
    srcs = [s for s, r in rates.items() if r]
    for s in srcs:
        head += "{:>16}".format(s[:14])
    print(head)
    for name, f in pairs:
        row = "    {:<28}{:>12}".format(name[:27], f"{f * 100:.2f}%")
        per = {}
        for s in srcs:
            aps = f * rates[s]["mean_frames_per_second"]
            per[s] = aps
            row += "{:>16}".format(f"{aps:.1f}/s")
        out["alarms_per_second"][name] = {"fpr": f, "per_capture": per}
        print(row)

    print("\n  Read this as: an alarm rate of r on a bus carrying v frames per")
    print("  second produces r*v alarms per second. The capture's own rate is")
    print("  the one measured above; a production bus may differ, and the")
    print("  conversion is linear either way.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n  wrote", args.out)


if __name__ == "__main__":
    main()
