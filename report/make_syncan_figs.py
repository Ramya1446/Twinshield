# -*- coding: utf-8 -*-
"""
Figures and tables for the second-dataset (SynCAN) experiment.

Reads the files the SynCAN runs wrote:

    results/syncan_results.json              closed-set run, both ID variants
    results/syncan_zero_day_dos.json         leave-one-attack-out, flooding held out
    results/syncan_zero_day_brake_tampering.json       plateau held out
    results/syncan_zero_day_steering_tampering.json    continuous held out

and writes figures/fig_syncan_zeroday.png, figures/fig_syncan_fpr.png plus the
table bodies, printed to stdout, that go into the paper and the report.

Slot names in those JSON files are this project's; each one stands for a SynCAN
attack. The mapping is read from the converted dataset's conversion.json and
every label this script emits is the SynCAN name, never the GEM one.

    python make_syncan_figs.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Three baselines read as one family on a neutral ramp; the proposed system is
# the only saturated hue on the axis, so it separates without relying on colour.
BASE = ["#c9c7c0", "#9b9992", "#6f6e6a"]

MODELS = ["CNN", "BiLSTM", "CNN-BiLSTM", "TwinShield-Fusion"]
SLOT_TO_SYNCAN = {"DoS": "flooding",
                  "Brake Tampering": "plateau",
                  "Steering Tampering": "continuous"}
SLUG = {"DoS": "dos", "Brake Tampering": "brake_tampering",
        "Steering Tampering": "steering_tampering"}


def load(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"missing: {path}\nRun the SynCAN experiment first; this "
                         f"script never invents a number for a run that is absent.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def alarm_rates(zd: dict, slot: str):
    """Detection rate on the quarantined class and the normal false-alarm rate."""
    out = {}
    for name in MODELS:
        m = zd["models"][name]
        out[name] = (100.0 * m["unknown_detection_rate"],
                     100.0 * m["false_positive_rate"],
                     m["known_class_metrics"]["Macro F1"])
    return out


def fig_zeroday(data, out_dir):
    held = list(data)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    n = len(MODELS)
    width = 0.78 / n
    for k, name in enumerate(MODELS):
        xs = [i + (k - (n - 1) / 2) * width for i in range(len(held))]
        ys = [data[h][name][0] for h in held]
        colour = BLUE if name == "TwinShield-Fusion" else BASE[k]
        ax.bar(xs, ys, width=width * 0.92, color=colour,
               edgecolor=INK if name == "TwinShield-Fusion" else GREY,
               linewidth=0.6, label=name, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2.0, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=6.6, color=INK, zorder=4)
    ax.set_xticks(range(len(held)))
    ax.set_xticklabels([f"{SLOT_TO_SYNCAN[h]}\nheld out" for h in held])
    ax.set_ylabel("frames of the held-out attack\nthat raised an alarm (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color=GRIDC, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
              frameon=False, handlelength=1.1, columnspacing=1.4)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_syncan_zeroday.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_fpr(data, out_dir):
    held = list(data)
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    n = len(MODELS)
    width = 0.78 / n
    for k, name in enumerate(MODELS):
        xs = [i + (k - (n - 1) / 2) * width for i in range(len(held))]
        ys = [data[h][name][1] for h in held]
        colour = ORANGE if name == "TwinShield-Fusion" else BASE[k]
        ax.bar(xs, ys, width=width * 0.92, color=colour,
               edgecolor=INK if name == "TwinShield-Fusion" else GREY,
               linewidth=0.6, label=name, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y, f" {y:.2f}", ha="center", va="bottom",
                    fontsize=6.6, color=INK, zorder=4)
    ax.set_xticks(range(len(held)))
    ax.set_xticklabels([f"{SLOT_TO_SYNCAN[h]}\nheld out" for h in held])
    ax.set_ylabel("false alarms on normal\ntraffic (%)")
    ax.yaxis.grid(True, color=GRIDC, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
              frameon=False, handlelength=1.1, columnspacing=1.4)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_syncan_fpr.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conv = os.path.join(ROOT, "data", "SynCAN_as_GEM", "conversion.json")
    if os.path.exists(conv):
        print("slot assignment:", json.load(open(conv))["slot_assignment"], "\n")

    # ------------------------------------------------ closed-set comparison
    res = load(os.path.join(args.results, "syncan_results.json"))
    print("TABLE  closed-set macro-F1 on SynCAN")
    print(f"{'Model':<34}{'with ID':>10}{'without ID':>13}")
    for name in MODELS + ["GRU Digital Twin + CNN-BiLSTM"]:
        row = []
        for v in ("with_id", "no_id"):
            m = res["results"].get(v, {}).get(name)
            row.append(f"{m['metrics']['macro_f1']:.4f}" if m else "--")
        print(f"{name:<34}{row[0]:>10}{row[1]:>13}")
    dt = res["digital_twin"]
    print(f"\ntwin on held-out normal SynCAN frames: MAE {dt['MAE']:.4f} bytes "
          f"({dt['MAE_normalised']:.5f} normalised), {dt['n_val_normal_frames']:,} frames")

    # ------------------------------------------------ leave-one-attack-out
    data = {}
    for slot in ("DoS", "Brake Tampering", "Steering Tampering"):
        zd = load(os.path.join(args.results, f"syncan_zero_day_{SLUG[slot]}.json"))
        data[slot] = alarm_rates(zd, slot)
        g = zd["gate"]
        print(f"\nTABLE  leave-one-attack-out, SynCAN {SLOT_TO_SYNCAN[slot]} held out"
              f"   (gate threshold z = {g['threshold']:.4f})")
        print(f"{'Model':<24}{'detected %':>12}{'false alarms %':>16}"
              f"{'known macro-F1':>17}")
        for name in MODELS:
            d, f, mf = data[slot][name]
            print(f"{name:<24}{d:>12.2f}{f:>16.2f}{mf:>17.4f}")
        st = g.get("score_stats", {}).get(slot)
        if st:
            print(f"    channel: frame z {st['mean_frame_z']:.3f}  "
                  f"ecu z {st['mean_ecu_z']:.3f}  "
                  f"frame share {st['channel_frame_share']:.2f}")

    print("\n ", fig_zeroday(data, args.out))
    print(" ", fig_fpr(data, args.out))


if __name__ == "__main__":
    main()
