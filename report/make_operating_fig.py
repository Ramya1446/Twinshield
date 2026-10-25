# -*- coding: utf-8 -*-
"""
Detection rate against false-alarm rate as the gate threshold is swept, drawn
from the sweep each leave-one-attack-out run records. The operating point the
paper reports is marked, so it can be read against the alternatives rather
than presented as the only possible choice.

    python make_operating_fig.py
"""
from __future__ import annotations

import argparse
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is not installed.\n\n    pip install matplotlib\n")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.8,
})
INK, GREY, GRIDC = "#111111", "#52514e", "#e1e0d9"
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"

# the three curves coincide, so they are drawn with different dash patterns
# and widths; the caption says they overlap rather than leaving it to be guessed
RUNS = [("ciciov_zero_day_dos.json", "DoS", "DoS", BLUE, "-", 3.2),
        ("ciciov_zero_day_brake_tampering.json", "Brake Tampering",
         "powertrain spoofing", AQUA, (0, (5, 2)), 2.0),
        ("ciciov_zero_day_steering_tampering.json", "Steering Tampering",
         "steering spoofing", ORANGE, (0, (1.5, 1.8)), 1.3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures",
                                                  "fig_operating_point.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for fname, slot, label, colour, dash, lw in RUNS:
        p = os.path.join(args.results, fname)
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}\nRun the experiment first; this script "
                             f"never invents a curve for a run that is absent.")
        with open(p, encoding="utf-8") as fh:
            run = json.load(fh)
        sw = run["gate"]["sweep"]
        fpr = np.asarray(sw["fpr"]) * 100.0
        det = np.asarray(sw[slot]) * 100.0
        order = np.argsort(fpr)
        auc = run["gate"]["auc_vs_normal"][slot]
        ax.plot(fpr[order], det[order], color=colour, linewidth=lw, linestyle=dash,
                label=f"{label}  (AUC {auc:.3f})", zorder=3, solid_capstyle="butt")
        op = run["models"]["TwinShield-Fusion"]
        ax.plot(op["false_positive_rate"] * 100.0,
                op["unknown_detection_rate"] * 100.0, "o", markersize=5.5,
                markerfacecolor="white", markeredgecolor=INK,
                markeredgewidth=1.0, zorder=5)

    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 104)
    ax.set_xlabel("false alarms on normal traffic (%)")
    ax.set_ylabel("held-out class detected (%)")
    ax.set_xticks([0, 0.05, 0.5, 5, 50])
    ax.set_xticklabels(["0", "0.05", "0.5", "5", "50"])
    ax.grid(True, color=GRIDC, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    ax.annotate("operating point\nreported in Table VIII", (0.37, 100),
                textcoords="offset points", xytext=(0, -16), ha="center",
                va="top", fontsize=6.4, color=GREY)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(args.out)
    for fname, slot, label, *_ in RUNS:
        with open(os.path.join(args.results, fname), encoding="utf-8") as fh:
            run = json.load(fh)
        sw = run["gate"]["sweep"]
        fpr = np.asarray(sw["fpr"]); det = np.asarray(sw[slot])
        keep = det >= 1.0
        best = fpr[keep].min() * 100 if keep.any() else float("nan")
        print(f"  {label:<22} AUC {run['gate']['auc_vs_normal'][slot]:.4f}   "
              f"100% detection still holds at {best:.3f}% false alarms")


if __name__ == "__main__":
    main()
