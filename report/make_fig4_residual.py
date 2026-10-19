# -*- coding: utf-8 -*-
"""
Figure 4 of the paper: mean digital-twin residual on the two tampering
identifiers, drawn from results/gem_zero_day.json so the picture cannot
disagree with the run that produced it.

    python make_fig4_residual.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
})
INK, GREY, GRIDC = "#111111", "#52514e", "#e1e0d9"
AQUA, ORANGE = "#1baf7a", "#eb6834"      # normal traffic / attack traffic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-day", default=os.path.join(ROOT, "results",
                                                       "gem_zero_day.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures",
                                                  "fig4_residual.png"))
    args = ap.parse_args()
    if not os.path.exists(args.zero_day):
        raise SystemExit(f"missing: {args.zero_day}\nThis figure is drawn from a run, "
                         f"never from remembered numbers.")
    with open(args.zero_day, encoding="utf-8") as fh:
        rows = json.load(fh)["gate"]["residual_by_can_id"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    w = 0.34
    for i, r in enumerate(rows):
        n, a = r["mean_residual_normal"], r["mean_residual_attack"]
        ax.bar(i - w / 2, n, width=w, color=AQUA, edgecolor=INK, linewidth=0.5,
               zorder=3, label="Normal frames on this ID" if i == 0 else None)
        ax.bar(i + w / 2, a, width=w, color=ORANGE, edgecolor=INK, linewidth=0.5,
               zorder=3, label="Attack frames on this ID" if i == 0 else None)
        ax.text(i - w / 2, n + 0.6, f"{n:.2f}", ha="center", va="bottom",
                fontsize=7, color=INK, zorder=4)
        ax.text(i + w / 2, a + 0.6, f"{a:.2f}", ha="center", va="bottom",
                fontsize=7, color=INK, zorder=4)
        ratio = a / n
        tag = f"{ratio:.1f}x rise" if ratio >= 2 else f"{ratio:.2f}x, no rise"
        ax.text(i, max(n, a) + 3.4, tag, ha="center", va="bottom", fontsize=7.5,
                color=ORANGE if ratio >= 2 else GREY, zorder=4)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['can_id']}\n({r['class']})" for r in rows], fontsize=7.5)
    ax.set_ylabel("mean |residual| (bytes)")
    ax.set_ylim(0, max(r["mean_residual_attack"] for r in rows) * 1.42)
    ax.yaxis.grid(True, color=GRIDC, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(args.out)
    for r in rows:
        print(f"  {r['can_id']}  {r['class']:<20} {r['mean_residual_normal']:.4f} -> "
              f"{r['mean_residual_attack']:.4f}  "
              f"({r['mean_residual_attack'] / r['mean_residual_normal']:.2f}x)")


if __name__ == "__main__":
    main()
