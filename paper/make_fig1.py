# -*- coding: utf-8 -*-
"""
Figure 1 of the IEEE Access paper: the TwinShield-Fusion architecture.

Sized for a single IEEE column (3.2 in), so the layout is vertical with two
parallel branches rather than wide. Each box keeps the Section IV sub-section
that specifies it, which the caption promises. Labels are measured against the
box that holds them and shrunk to fit, so nothing overflows at column width.

    python make_fig1.py <output-dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
})
INK, GREY, SLATE = "#111111", "#52514e", "#6f6e6a"
BLUE, BLUEF = "#2a78d6", "#dfeafa"
AQUA, AQUAF = "#1baf7a", "#ddf2e9"
ORANGE, ORANGEF = "#eb6834", "#fce7dd"
LIGHT = "#FFFFFF"
EDGE_OF = {BLUEF: BLUE, AQUAF: AQUA, ORANGEF: ORANGE, LIGHT: SLATE}

fig, ax = plt.subplots(figsize=(3.3, 4.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")

_R = {"r": None}


def fit(xc, yc, text, box_w, fs, **kw):
    """Shrink text until it fits box_w, measured rather than estimated."""
    t = ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, **kw)
    if _R["r"] is None:
        fig.canvas.draw(); _R["r"] = fig.canvas.get_renderer()
    p0, p1 = ax.transData.transform([(0, 0), (box_w, 0)])
    avail = (p1[0] - p0[0]) * 0.90
    for _ in range(12):
        w = t.get_window_extent(_R["r"]).width
        if w <= avail or fs <= 2.8:
            break
        fs = max(2.8, fs * avail / w)
        t.set_fontsize(fs)
    return t


def box(x, y, w, h, title, sub=None, sec=None, fill=LIGHT, fs=5.4, sfs=4.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.25,rounding_size=1.1",
                 linewidth=0.85, edgecolor=EDGE_OF[fill], facecolor=fill))
    if sub:
        fit(x + w / 2, y + h * 0.63, title, w, fs, weight="bold", color=INK)
        fit(x + w / 2, y + h * 0.28, sub, w, sfs, color=GREY, style="italic")
    else:
        fit(x + w / 2, y + h / 2, title, w, fs, weight="bold", color=INK)
    if sec:
        ax.text(x + w - 0.6, y + h + 0.7, sec, ha="right", va="bottom",
                fontsize=4.0, weight="bold", color=EDGE_OF[fill])


def arr(x1, y1, x2, y2, lw=0.8, color=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=5.5, linewidth=lw, linestyle=ls,
                                 color=color or SLATE, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ source
box(16, 120, 68, 10, "CAN bus traffic", "ID, DLC, 8 payload bytes, timing",
    sec="IV-B", fill=AQUAF)
ax.plot([50, 50], [120, 115.5], color=SLATE, lw=0.8)
ax.plot([24, 76], [115.5, 115.5], color=SLATE, lw=0.8)
arr(24, 115.5, 24, 110)
arr(76, 115.5, 76, 110)

L, R, BW = 2, 52, 46                      # two branch columns

# -------------------------------------------------- unsupervised, left
for i, (t, sb, sc) in enumerate([
        ("Per-ID history", "last 8 payloads", "IV-C"),
        ("GRU digital twin", "normal traffic only", "IV-C"),
        ("Residual  r_t", "|actual − predicted|", "IV-C"),
        ("Standardise", "per ID, per byte", "IV-F"),
        ("Threshold  z > z*", "calibrated FPR", "IV-F")]):
    y = 99 - i * 18.5
    box(L, y, BW, 11, t, sb, sec=sc, fill=BLUEF)
    if i < 4:
        arr(L + BW / 2, y, L + BW / 2, y - 7.5)

# ---------------------------------------------------- supervised, right
for i, (t, sb, sc) in enumerate([
        ("Context window", "32 frames", "IV-B"),
        ("CNN-BiLSTM", "convolution, then BiLSTM", "IV-D"),
        ("Adaptive ECU fusion", "attention over streams", "IV-E"),
        ("Known-class head", "softmax over k classes", "IV-D")]):
    y = 99 - i * 18.5
    box(R, y, BW, 11, t, sb, sec=sc, fill=AQUAF)
    if i < 3:
        arr(R + BW / 2, y, R + BW / 2, y - 7.5)

# the residual is also an input to the fusion stage
ax.plot([L + BW, 49.5], [67.5, 67.5], color=BLUE, lw=0.7, ls=(0, (2.5, 2)))
arr(49.5, 67.5, R, 67.5, lw=0.7, color=BLUE, ls=(0, (2.5, 2)))

# label the two branches where they split, clear of every box and tag
ax.text(22.5, 112.6, "unsupervised", fontsize=4.2, color=BLUE, style="italic",
        ha="right", va="center")
ax.text(77.5, 112.6, "supervised", fontsize=4.2, color=AQUA, style="italic",
        ha="left", va="center")

# ------------------------------------------------------------- decision
arr(L + BW / 2, 25, L + BW / 2, 18.5)
ax.plot([R + BW / 2, R + BW / 2], [43.5, 21.8], color=SLATE, lw=0.8)
ax.plot([R + BW / 2, L + BW / 2 + 14], [21.8, 21.8], color=SLATE, lw=0.8)
arr(L + BW / 2 + 14, 21.8, L + BW / 2 + 14, 18.5)

box(16, 8, 68, 10.5, "Alarm decision", "fires when either channel fires",
    sec="IV-G", fill=ORANGEF)
ax.text(50, 3.4, "named class  ·  or  ·  unknown anomaly", ha="center",
        fontsize=4.4, color=GREY, style="italic")

fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=600,
            bbox_inches="tight", facecolor="white", pad_inches=0.02)
print("wrote", os.path.join(OUT, "fig1_architecture.png"))
