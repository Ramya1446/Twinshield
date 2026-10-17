# -*- coding: utf-8 -*-
"""
Figures 4.1 and 4.2 of the report: the layered architecture and the end-to-end
pipeline.

    python make_design_figs.py figures

Both use one palette with one meaning per hue, the same as every other figure in
the report: aqua for data, blue for the components contributed by this work,
orange for the alarm they produce, slate for scaffolding. The three hues were
checked over all pairs against colour-vision deficiency (worst deuteranope
separation dE 9.2, normal vision 24.0), and every box carries its own text, so
colour is never the only channel carrying meaning.
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
BLUE, BLUEF = "#2a78d6", "#dfeafa"
AQUA, AQUAF = "#1baf7a", "#ddf2e9"
ORANGE, ORANGEF = "#eb6834", "#fce7dd"
LIGHT = "#FFFFFF"
EDGE_OF = {BLUEF: BLUE, AQUAF: AQUA, ORANGEF: ORANGE, LIGHT: SLATE}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print("wrote", p)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=lw, linestyle=ls,
                                 color=color or SLATE, shrinkA=0, shrinkB=0))


# ===================================================== Figure 4.1
def fig41():
    """Five layers, each naming its own modules, with the dependency direction
    drawn and the label-free / label-dependent split called out."""
    fig, ax = plt.subplots(figsize=(6.0, 4.15))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11.3); ax.axis("off")

    layers = [
        ("L5", "Presentation layer", "Streamlit pages: Overview  ·  Live Monitor  ·  Results",
         "pages/", LIGHT),
        ("L4", "Application layer", "replay controller  ·  cached model access  ·  figures",
         "ui/", LIGHT),
        ("L3", "Detection layer", "CNN  ·  BiLSTM  ·  CNN-BiLSTM  ·  TwinShield-Fusion  ·  anomaly gate",
         "models/, evaluation/", BLUEF),
        ("L2", "Modelling layer", "GRU digital twin  ·  per-byte residual  ·  adaptive ECU fusion",
         "models/digital_twin.py", BLUEF),
        ("L1", "Data layer", "GEM-CAN loader  ·  features  ·  context windows  ·  block split",
         "preprocessing/", AQUAF),
    ]
    x0, w, h, gap = 1.05, 7.6, 1.52, 0.52
    wide_gap = 1.25                     # room for the split annotation, L3 to L2
    ys, y = [], 9.6
    for i in range(len(layers)):
        ys.append(y)
        y -= h + (wide_gap if i == 2 else gap)

    for i, (tag, name, sub, mod, fill) in enumerate(layers):
        yy = ys[i]
        ax.add_patch(FancyBboxPatch((x0, yy), w, h,
                     boxstyle="round,pad=0.04,rounding_size=0.12",
                     linewidth=1.1, edgecolor=EDGE_OF[fill], facecolor=fill))
        ax.text(x0 + w / 2, yy + h * 0.66, name, ha="center", va="center",
                fontsize=9.0, weight="bold", color=INK)
        ax.text(x0 + w / 2, yy + h * 0.33, sub, ha="center", va="center",
                fontsize=6.7, color=GREY, style="italic")
        ax.text(x0 - 0.18, yy + h / 2, tag, ha="right", va="center",
                fontsize=7.4, weight="bold", color=EDGE_OF[fill])
        ax.text(x0 + w + 0.18, yy + h / 2, mod, ha="left", va="center",
                fontsize=6.0, color=GREY, family="monospace")
        if i < len(layers) - 1:
            arrow(ax, x0 + w / 2, yy, x0 + w / 2, ys[i + 1] + h + 0.04, lw=1.0)

    # "depends on", centred in the first gap so it clears both boxes
    ymid = (ys[0] + ys[1] + h) / 2
    ax.text(x0 + w / 2 + 0.28, ymid, "depends on", fontsize=6.2, color=GREY,
            style="italic", ha="left", va="center")

    # the architecturally significant split, in the wide gap between L3 and L2
    ysplit = (ys[2] + ys[3] + h) / 2
    ax.plot([0.35, 9.65], [ysplit, ysplit], color=SLATE, lw=0.9, ls=(0, (5, 3)))
    ax.text(0.35, ysplit + 0.17, "above: fitted against attack labels",
            fontsize=6.5, color=BLUE, style="italic", va="bottom", ha="left")
    ax.text(0.35, ysplit - 0.17, "below: fitted without any attack label",
            fontsize=6.5, color=AQUA, style="italic", va="top", ha="left")
    save(fig, "fig4_1_architecture.png")


# ===================================================== Figure 4.2
def fig42():
    """Seven stages, banded by the layer each belongs to, with the branch at the
    residual drawn rather than flattened into a single chain. Section
    references sit inside the boxes so no label can collide with a connector."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11.6); ax.axis("off")

    bands = [("DATA LAYER", 8.75, 11.5, AQUAF, AQUA),
             ("MODELLING LAYER", 6.05, 8.6, BLUEF, BLUE),
             ("DETECTION LAYER", 0.1, 5.9, BLUEF, BLUE)]
    for name, lo, hi, fill, edge in bands:
        ax.add_patch(FancyBboxPatch((0.05, lo), 9.9, hi - lo,
                     boxstyle="round,pad=0.0,rounding_size=0.06",
                     linewidth=0.0, facecolor=fill, alpha=0.30))
        ax.text(0.22, hi - 0.14, name, ha="left", va="top", fontsize=6.3,
                weight="bold", color=edge)

    bw, bh = 2.75, 1.78

    def stage(x, y, n, name, sub, sec, fill):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.04,rounding_size=0.12",
                     linewidth=1.1, edgecolor=EDGE_OF[fill], facecolor=fill))
        ax.text(x + 0.16, y + bh - 0.16, str(n), ha="left", va="top",
                fontsize=7.4, weight="bold", color=EDGE_OF[fill])
        ax.text(x + bw - 0.16, y + bh - 0.16, sec, ha="right", va="top",
                fontsize=6.2, color=GREY)
        ax.text(x + bw / 2, y + bh * 0.46, name, ha="center", va="center",
                fontsize=8.2, weight="bold", color=INK)
        ax.text(x + bw / 2, y + bh * 0.20, sub, ha="center", va="center",
                fontsize=6.4, color=GREY, style="italic")

    # band 1 -- data
    xs = [0.45, 3.62, 6.80]
    y1 = 8.95
    for i, (n, nm, sb, sc) in enumerate([
            (1, "CAN frames", "raw bus traffic", "\u00a75.2"),
            (2, "Features", "bytes, DLC, gap", "\u00a75.3"),
            (3, "Context window", "32 frames", "\u00a75.4")]):
        stage(xs[i], y1, n, nm, sb, sc, AQUAF)
        if i < 2:
            arrow(ax, xs[i] + bw, y1 + bh / 2, xs[i + 1], y1 + bh / 2, lw=1.1)

    # band 2 -- the single modelling stage
    y2 = 6.45
    stage(3.62, y2, 4, "Twin residual", "|actual \u2212 expected|", "\u00a75.5\u20135.6", BLUEF)
    ax.plot([8.18, 8.18], [y1, 8.68], color=SLATE, lw=1.1)
    ax.plot([8.18, 5.00], [8.68, 8.68], color=SLATE, lw=1.1)
    arrow(ax, 5.00, 8.68, 5.00, y2 + bh, lw=1.1)

    # band 3 -- the branch
    y3 = 3.30
    stage(0.45, y3, 5, "ECU fusion", "attention over streams", "\u00a75.7", BLUEF)
    stage(3.62, y3, 6, "Detector head", "softmax over classes", "\u00a75.9", BLUEF)
    stage(6.80, y3, 6, "Anomaly gate", "z > z*, normal-only", "\u00a75.10", BLUEF)
    y4 = 0.45
    stage(3.62, y4, 7, "Alarm", "class, or UNKNOWN", "\u00a75.13", ORANGEF)

    ybus = 6.08
    ax.plot([5.00, 5.00], [y2, ybus], color=SLATE, lw=1.1)
    ax.plot([1.82, 8.18], [ybus, ybus], color=SLATE, lw=1.1)
    for x in (1.82, 5.00, 8.18):
        arrow(ax, x, ybus, x, y3 + bh, lw=1.1)
    arrow(ax, 0.45 + bw, y3 + bh / 2, 3.62, y3 + bh / 2, lw=1.1)

    # both channels converge on the alarm
    arrow(ax, 5.00, y3, 5.00, y4 + bh, lw=1.1)
    ax.plot([8.18, 8.18], [y3, 2.78], color=SLATE, lw=1.1)
    ax.plot([8.18, 6.10], [2.78, 2.78], color=SLATE, lw=1.1)
    arrow(ax, 6.10, 2.78, 6.10, y4 + bh, lw=1.1)

    # the note sits in the empty column under stage 5, clear of every connector
    ax.text(0.45, 1.34, "Stage 6 runs two\nparallel channels:\nthe head names a\n"
            "class it was trained\non, the gate needs\nno label at all.",
            ha="left", va="center", fontsize=6.4, color=GREY, style="italic",
            linespacing=1.5)
    save(fig, "fig4_2_pipeline.png")


fig41()
fig42()
