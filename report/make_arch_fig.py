# -*- coding: utf-8 -*-
"""
Figure 4.1, the TwinShield-Fusion architecture.

Follows the two-branch layout the author supplied, redrawn at 300 dpi in the
report's serif face, with the component names corrected to match Chapter 5:
the classifier is the CNN-BiLSTM actually used rather than a generic choice,
the adaptive multi-ECU residual fusion stage is present, the 32-frame context
window is named, and the decision carries the three outcomes the system can
actually report.

Every label is sized against the width of the box that holds it at the printed
figure width of 5.7 in, so nothing overflows once Word scales the image.

    python make_arch_fig.py figures
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
})

INK, GREY, SLATE = "#111111", "#52514e", "#6f6e6a"
BLUE,  BLUEF  = "#2a78d6", "#dfeafa"       # the digital-twin branch
AQUA,  AQUAF  = "#1baf7a", "#ddf2e9"       # data and the supervised branch
ORANGE, ORANGEF = "#eb6834", "#fce7dd"     # the alarm
PALE, HAIR = "#f4f4f1", "#d7d6d0"

W_IN, H_IN = 6.6, 3.55
fig, ax = plt.subplots(figsize=(W_IN, H_IN))
ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")


# --- text that is measured against the box it sits in -------------------
_RENDERER = {"r": None}


def _renderer():
    if _RENDERER["r"] is None:
        fig.canvas.draw()
        _RENDERER["r"] = fig.canvas.get_renderer()
    return _RENDERER["r"]


def fit(xc, yc, text, box_w, fs, **kw):
    """Place text centred at (xc, yc) and shrink it until it fits box_w units.

    Guessing point sizes against box widths is what made the first draft of
    this figure overflow, so the width is measured rather than estimated.
    """
    t = ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, **kw)
    p0, p1 = ax.transData.transform([(0, 0), (box_w, 0)])
    avail = (p1[0] - p0[0]) * 0.88
    for _ in range(12):
        w = t.get_window_extent(_renderer()).width
        if w <= avail or fs <= 3.4:
            break
        fs = max(3.4, fs * avail / w)
        t.set_fontsize(fs)
    return t


def rect(x, y, w, h, fill, edge, lw=1.0, r=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.2,rounding_size={r}",
                 linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls))


def stage(x, y, w, h, title, sub, fill, edge, fs=6.0, sfs=5.0):
    rect(x, y, w, h, fill, edge)
    if sub:
        fit(x + w / 2, y + h * 0.62, title, w, fs, weight="bold", color=INK)
        fit(x + w / 2, y + h * 0.27, sub, w, sfs, color=GREY, style="italic")
    else:
        fit(x + w / 2, y + h / 2, title, w, fs, weight="bold", color=INK)


def chip(x, y, w, h, text, fs=4.9):
    rect(x, y, w, h, PALE, HAIR, lw=0.7, r=0.7)
    fit(x + w / 2, y + h / 2, text, w, fs, color=INK)


def arr(x1, y1, x2, y2, lw=0.9, color=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=6.5, linewidth=lw, linestyle=ls,
                                 color=color or SLATE, shrinkA=0, shrinkB=0))


def container(x, y, w, h, title, edge):
    rect(x, y, w, h, edge, edge, lw=0.0, r=1.3)
    ax.patches[-1].set_alpha(0.07)
    rect(x, y, w, h, "none", edge, lw=0.9, r=1.3, ls=(0, (4, 2.5)))
    ax.text(x + 1.2, y + h - 1.9, title, ha="left", va="center",
            fontsize=6.0, weight="bold", color=edge)


# ----------------------------------------------------------------- source
ax.text(4.0, 30.0, "CAN bus", ha="center", fontsize=6.6, weight="bold", color=INK)
ax.text(4.0, 26.8, "(from vehicle)", ha="center", fontsize=5.2, color=GREY,
        style="italic")
arr(8.3, 28.4, 11.8, 28.4)

# ----------------------------------------------------------- preprocessing
rect(11.8, 15.5, 14.4, 26.0, AQUAF, AQUA)
fit(19.0, 39.3, "CAN frame", 14.4, 6.2, weight="bold", color=INK)
fit(19.0, 36.6, "preprocessing", 14.4, 6.2, weight="bold", color=INK)
for i, t in enumerate(["Arbitration ID", "Data length code",
                       "Payload bytes 0–8", "Timestamp"]):
    chip(13.3, 30.6 - i * 4.0, 11.4, 3.1, t)

# branch out
ax.plot([26.2, 28.6], [28.4, 28.4], color=SLATE, lw=0.9)
ax.plot([28.6, 28.6], [11.5, 44.5], color=SLATE, lw=0.9)
arr(28.6, 44.5, 31.5, 44.5)
arr(28.6, 11.5, 31.5, 11.5)

# ------------------------------------------------------------ digital twin
container(31.5, 35.0, 40.0, 17.5, "Digital twin — model of normal behaviour", BLUE)
BW, BH = 11.2, 10.0
for i, (ttl, sub) in enumerate([("Per-ID history", "last 8 payloads"),
                                ("GRU", "normal traffic only"),
                                ("Residual", "|actual − predicted|")]):
    x = 33.0 + i * 12.6
    stage(x, 37.5, BW, BH, ttl, sub, BLUEF, BLUE)
    if i < 2:
        arr(x + BW, 42.5, x + 12.6, 42.5)

# -------------------------------------------------------- supervised branch
container(31.5, 2.0, 40.0, 17.5, "Supervised classifier — known attacks", AQUA)
for i, (ttl, sub) in enumerate([("Context window", "32 frames"),
                                ("CNN-BiLSTM", "+ adaptive ECU fusion"),
                                ("Class posterior", "softmax, k classes")]):
    x = 33.0 + i * 12.6
    stage(x, 4.5, BW, BH, ttl, sub, AQUAF, AQUA)
    if i < 2:
        arr(x + BW, 9.5, x + 12.6, 9.5)

# the residual is also an input to the fusion stage: it drops straight down
# the middle rather than looping round the preprocessing block, so it cannot be
# mistaken for an arrow into the context window
XR = 51.2
ax.plot([63.8, 63.8], [37.5, 27.5], color=BLUE, lw=0.8, ls=(0, (3, 2)))
ax.plot([63.8, XR], [27.5, 27.5], color=BLUE, lw=0.8, ls=(0, (3, 2)))
arr(XR, 27.5, XR, 14.5, lw=0.8, color=BLUE, ls=(0, (3, 2)))
ax.text(XR + 0.9, 21.0, "residual streams", fontsize=4.8, color=BLUE,
        ha="left", va="center", style="italic")

# ------------------------------------------------------------ anomaly gate
rect(75.5, 33.5, 23.0, 14.5, BLUEF, BLUE)
fit(87.0, 44.2, "Residual anomaly gate", 23.0, 6.2, weight="bold", color=INK)
fit(87.0, 40.6, "standardise per ID and per byte", 23.0, 5.0, color=GREY, style="italic")
fit(87.0, 37.2, "alarm when z > z*, calibrated FPR", 23.0, 5.0, color=GREY, style="italic")
arr(70.4, 42.5, 75.5, 42.0)

# ---------------------------------------------------------- alarm decision
stage(75.5, 19.5, 23.0, 9.0, "Alarm decision", "fires if either channel fires",
      ORANGEF, ORANGE, fs=6.2, sfs=5.0)
arr(87.0, 33.5, 87.0, 28.5)
ax.text(87.8, 31.0, "anomaly score", fontsize=4.9, color=GREY, ha="left",
        va="center", style="italic")
ax.plot([70.4, 73.2], [9.5, 9.5], color=SLATE, lw=0.9)
ax.plot([73.2, 73.2], [9.5, 24.0], color=SLATE, lw=0.9)
arr(73.2, 24.0, 75.5, 24.0)
ax.text(73.9, 13.5, "class score", fontsize=4.9, color=GREY, ha="left",
        va="center", style="italic")

# ----------------------------------------------------------------- outputs
for x, t, f, e in [(75.5, "Normal", AQUAF, AQUA),
                   (83.2, "Known class", AQUAF, AQUA),
                   (90.9, "Unknown", ORANGEF, ORANGE)]:
    stage(x, 2.5, 7.1, 5.6, t, None, f, e, fs=5.4)
ax.plot([87.0, 87.0], [19.5, 12.0], color=SLATE, lw=0.9)
ax.plot([79.0, 94.5], [12.0, 12.0], color=SLATE, lw=0.9)
for x in (79.0, 86.75, 94.5):
    arr(x, 12.0, x, 8.1)

fig.savefig(os.path.join(OUT, "fig4_1_architecture.png"), dpi=300,
            bbox_inches="tight", facecolor="white", pad_inches=0.04)
print("wrote", os.path.join(OUT, "fig4_1_architecture.png"))
