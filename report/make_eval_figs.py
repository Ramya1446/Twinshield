# -*- coding: utf-8 -*-
"""
Build the classifier-diagnostic figures for the report: a confusion matrix,
one-vs-rest ROC curves and precision-recall curves.

Everything is computed from the test-split probabilities your own training run
saved, so the figures cannot disagree with the tables beside them.

    python make_eval_figs.py                      # ../saved_models -> ./figures
    python make_eval_figs.py --variant no_id      # the ID-free run
    python make_eval_figs.py --saved ../saved_models --out figures

The fast path reads saved_models/<variant>__twinshield_test_proba.npy, which
train.py writes for the proposed model. If that file is absent the script says
so and stops; it never substitutes a plausible-looking curve for a missing
measurement.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from preprocessing.loader import CLASS_NAMES          # noqa: E402
from preprocessing.prepare import prepare             # noqa: E402

# --------------------------------------------------------------------------
# Look and feel: the palette used by every other figure in the report.
# Each panel carries a single curve, so no two hues ever share an axis and the
# categorical pair checks do not apply; the sequential ramp below is one hue
# running light to dark, as a magnitude encoding requires.
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
})
W = 6.0
INK, GREY, GRIDC = "#111111", "#52514e", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# blue ramp, steps 100 -> 700, light to dark
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRIDC, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.04)
    plt.close(fig)
    print("  wrote", path)


def ramp_colour(frac):
    """Pick a step of the blue ramp for a value in [0, 1]."""
    i = int(round(frac * (len(BLUE_RAMP) - 1)))
    return BLUE_RAMP[max(0, min(len(BLUE_RAMP) - 1, i))]


# ==========================================================================
# Figure: confusion matrix
# ==========================================================================
def fig_confusion(y_true, y_pred, names, out, tag):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    row_tot = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, np.maximum(row_tot, 1))

    fig, ax = plt.subplots(figsize=(W * 0.78, W * 0.62))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.6, rotation=20, ha="right")
    ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")

    for i in range(len(names)):
        for j in range(len(names)):
            # the count is the measurement; the share is the context
            txt = f"{cm[i, j]:,}\n{frac[i, j]*100:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.0,
                    color="white" if frac[i, j] > 0.55 else INK,
                    weight="bold" if i == j else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    save(fig, out, f"fig6_6_confusion{tag}.png")

    print("\n  Confusion matrix (rows: true class, columns: predicted)")
    print("    {:<20}".format("") + "".join(f"{n[:10]:>12}" for n in names) + f"{'support':>10}")
    for i, n in enumerate(names):
        print("    {:<20}".format(n)
              + "".join(f"{cm[i, j]:>12,}" for j in range(len(names)))
              + f"{int(row_tot[i, 0]):>10,}")


# ==========================================================================
# Figures: one-vs-rest ROC and precision-recall, one panel per class
# ==========================================================================
def _panels(n):
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(W, 2.5 * rows))
    return fig, np.atleast_1d(axes).ravel()


def fig_roc(y_true, proba, names, out, tag):
    fig, axes = _panels(len(names))
    aucs = {}
    for k, name in enumerate(names):
        ax = axes[k]
        yk = (y_true == k).astype(int)
        if yk.sum() == 0:
            ax.axis("off"); continue
        fpr, tpr, _ = roc_curve(yk, proba[:, k])
        a = auc(fpr, tpr); aucs[name] = a
        ax.plot([0, 1], [0, 1], lw=1.0, ls=(0, (4, 3)), color=GREY)
        ax.plot(fpr, tpr, lw=2.0, color=BLUE)
        ax.fill_between(fpr, tpr, alpha=0.10, color=BLUE, linewidth=0)
        ax.set_title(f"{name}   AUC = {a:.4f}", fontsize=8.6, loc="left",
                     color=INK, weight="bold")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.annotate(f"n = {int(yk.sum()):,}", (0.97, 0.06), xycoords="axes fraction",
                    ha="right", fontsize=7.2, color=GREY)
        despine(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, out, f"fig6_7_roc{tag}.png")

    print("\n  One-vs-rest ROC AUC")
    for n, a in aucs.items():
        print(f"    {n:<22}{a:.4f}")


def fig_pr(y_true, proba, names, out, tag):
    fig, axes = _panels(len(names))
    aps = {}
    for k, name in enumerate(names):
        ax = axes[k]
        yk = (y_true == k).astype(int)
        if yk.sum() == 0:
            ax.axis("off"); continue
        prec, rec, _ = precision_recall_curve(yk, proba[:, k])
        ap = average_precision_score(yk, proba[:, k]); aps[name] = ap
        base = yk.mean()
        ax.axhline(base, lw=1.0, ls=(0, (4, 3)), color=GREY)
        ax.plot(rec, prec, lw=2.0, color=ORANGE)
        ax.fill_between(rec, prec, alpha=0.10, color=ORANGE, linewidth=0)
        ax.set_title(f"{name}   AP = {ap:.4f}", fontsize=8.6, loc="left",
                     color=INK, weight="bold")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        # sits on the baseline itself, so it never lands under the curve or
        # on top of the dashed line it is labelling
        ax.annotate(f"no-skill = {base:.3f}", (0.03, base), xycoords="data",
                    ha="left", va="bottom", fontsize=7.2, color=GREY,
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
        despine(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, out, f"fig6_8_pr{tag}.png")

    print("\n  One-vs-rest average precision")
    for n, a in aps.items():
        print(f"    {n:<22}{a:.4f}")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved", default=os.path.join(HERE, "..", "saved_models"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--variant", default="with_id", choices=["with_id", "no_id"])
    args = ap.parse_args()

    proba_path = os.path.join(args.saved, f"{args.variant}__twinshield_test_proba.npy")
    if not os.path.exists(proba_path):
        print(f"  [stop] {proba_path} not found.")
        print("  Run:  python train.py")
        return

    print(f"reading {proba_path}")
    proba = np.load(proba_path)

    p = prepare()
    test_idx = p.idx("test")
    y_true = p.windows.y[test_idx]
    if len(y_true) != len(proba):
        print(f"  [stop] {len(proba):,} saved probabilities against "
              f"{len(y_true):,} test frames. The split has changed since "
              f"train.py last ran; re-run it before plotting.")
        return

    names = list(CLASS_NAMES)[:proba.shape[1]]
    y_pred = proba.argmax(1)
    tag = "" if args.variant == "with_id" else "_no_id"

    print(f"\nTwinShield-Fusion, variant '{args.variant}', "
          f"{len(y_true):,} test frames, {len(names)} classes")
    fig_confusion(y_true, y_pred, names, args.out, tag)
    fig_roc(y_true, proba, names, args.out, tag)
    fig_pr(y_true, proba, names, args.out, tag)
    print("\nDone. Figures 6.6, 6.7 and 6.8 are in", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
