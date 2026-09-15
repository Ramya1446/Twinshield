"""
Page 2 -- Live Monitor.  The main judge-facing demo.

Replays real GEM-CAN frames one at a time: a lead-in from the normal-driving
capture followed by a slice of the attack capture (the two recordings are joined,
which the page states openly, because the attack capture is already under attack
from its first frame).  On every tick the newly revealed frames are pushed through
the trained detector with a real forward pass; the status banner, residual trace,
attention and pipeline readouts all display values returned by that inference.
Nothing is scripted.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from preprocessing.loader import CLASS_NAMES
from ui.components import flag, frame_rows, note, section, status_banner
from ui.data import (data_ready, frame_table, get_prepared, get_scenarios,
                     get_twin_outputs, id_stream_labels, infer, missing_notice,
                     models_ready)
from ui.theme import AMBER, BLUE, CYAN, GREEN, MUTED, RED, TEXT, style_fig

STEP = 3            # frames revealed per tick
TICK = 0.10         # seconds between ticks
TAIL = 160          # frames kept on the residual trace

# ---------------------------------------------------------------- header
h1, h2 = st.columns([2.4, 1])
with h1:
    st.markdown("## 🚗 Vehicle Security Monitor")
with h2:
    st.markdown(
        f'<div style="text-align:right;padding-top:.75rem">'
        f'<span class="ts-dot ts-live" style="background:{GREEN}"></span>'
        f'<b style="color:{GREEN};font-size:.9rem">SYSTEM ONLINE</b>'
        f'<div style="color:{MUTED};font-size:.8rem;margin-top:.15rem">'
        f'Vehicle: GEM e6 EV · CAN 250 kbps</div></div>', unsafe_allow_html=True)

if not data_ready():
    flag("⚠️ GEM-CAN dataset not found — see Overview.")
    st.stop()
if not models_ready():
    missing_notice()
    st.stop()

p = get_prepared()
scenarios = get_scenarios()
if not scenarios:
    flag("No attack scenarios found in the dataset.")
    st.stop()

# ---------------------------------------------------------------- controls
names = list(scenarios.keys())
c1, c2, c3, c4 = st.columns([1.7, 1, 1, 1])
with c1:
    scenario = st.selectbox("Attack Scenario", names, key="scenario",
                            label_visibility="collapsed")
with c2:
    play = st.button("▶  Replay Attack", type="primary", use_container_width=True)
with c3:
    pause = st.button("⏸  Pause", use_container_width=True)
with c4:
    reset = st.button("↻  Reset", use_container_width=True)

sc = scenarios[scenario]
frames = sc["frames"]
n_frames = len(frames)
lead_n = sc["lead"]

# ---------------------------------------------------------------- state
ss = st.session_state
if ss.get("sc_key") != scenario:
    ss.sc_key = scenario
    ss.cursor = 0
    ss.playing = False
    ss.proba = np.zeros((0, 4), dtype=np.float32)
    ss.alpha = np.zeros((0, len(id_stream_labels())), dtype=np.float32)

if play:
    ss.cursor, ss.playing = 0, True
    ss.proba = np.zeros((0, 4), dtype=np.float32)
    ss.alpha = np.zeros((0, len(id_stream_labels())), dtype=np.float32)
if pause:
    ss.playing = False
if reset:
    ss.cursor, ss.playing = 0, False
    ss.proba = np.zeros((0, 4), dtype=np.float32)
    ss.alpha = np.zeros((0, len(id_stream_labels())), dtype=np.float32)

# ---------------------------------------------------------------- inference
# Only the frames revealed on this tick are pushed through the model.
if ss.playing and ss.cursor < n_frames:
    lo, hi = ss.cursor, min(ss.cursor + STEP, n_frames)
    proba, alpha = infer(frames[lo:hi])
    ss.proba = np.concatenate([ss.proba, proba]) if len(ss.proba) else proba
    ss.alpha = np.concatenate([ss.alpha, alpha]) if len(ss.alpha) else alpha
    ss.cursor = hi
elif ss.cursor == 0 and not ss.playing:
    # Idle: show the first few frames so the console is never blank.
    n_seed = min(12, n_frames)
    proba, alpha = infer(frames[:n_seed])
    ss.proba, ss.alpha = proba, alpha
    ss.cursor = n_seed

shown = len(ss.proba)
idx = frames[:shown]
preds = ss.proba.argmax(1)
confs = ss.proba.max(1)
cur_cls = CLASS_NAMES[int(preds[-1])] if shown else "Normal"
cur_conf = float(confs[-1]) if shown else 0.0
# "Suspicious" = model still says Normal but its confidence has dropped.
suspicious = cur_cls == "Normal" and cur_conf < 0.90

# ---------------------------------------------------------------- status
phase = ("normal-driving capture" if shown <= lead_n else "attack capture")
status_banner(
    cur_cls, cur_conf,
    subtitle=(f"Frame {shown:,} / {n_frames:,} · now replaying the {phase} · "
              f"{sc['test_share']}% of these frames are held-out test data"),
    suspicious=suspicious)

st.write("")

# ---------------------------------------------------------------- live pipeline
residual, expected = get_twin_outputs()
cur_res = float(residual[idx[-1]].mean() * 255) if shown else 0.0
labels = id_stream_labels()
top_stream = labels[int(ss.alpha[-1].argmax())] if shown and ss.alpha.size else "—"

stages = [
    ("🚗", "CAN TRAFFIC", f"{shown} frames read"),
    ("🧠", "DIGITAL TWIN", "expected payload"),
    ("⚡", "RESIDUAL", f"{cur_res:.1f} bytes"),
    ("🔗", "ADAPTIVE FUSION", f"top ECU {top_stream}"),
    ("🤖", "CNN-BiLSTM", "forward pass"),
    ("🚨", "DETECTION", f"{cur_cls} {cur_conf*100:.0f}%"),
]
parts = ['<div class="ts-pipe">']
for i, (ico, nm, ds) in enumerate(stages):
    cls = "ts-stage on" if (ss.playing and i == len(stages) - 1) else "ts-stage done"
    parts.append(f'<div class="{cls}"><span class="ico">{ico}</span>'
                 f'<span class="nm">{nm}</span><span class="ds">{ds}</span></div>')
    if i < len(stages) - 1:
        parts.append('<div class="ts-arrow">→</div>')
parts.append("</div>")
st.markdown("".join(parts), unsafe_allow_html=True)
note("Each box shows a value produced by the model for the frame currently being "
     "processed — the readouts update because inference runs, not on a timer.")

st.write("")

# ---------------------------------------------------------------- stream + charts
left, right = st.columns([1.35, 1], gap="large")

with left:
    section("CAN Traffic Stream", small=True)
    if shown:
        table = frame_table(idx, preds, confs)
        frame_rows(table, limit=13)
    else:
        st.caption("Press ▶ Replay Attack to begin.")

with right:
    section("Digital Twin Residual", small=True)
    if shown:
        tail = slice(max(0, shown - TAIL), shown)
        res_series = residual[idx[tail]].mean(axis=1) * 255
        pr = preds[tail]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=res_series, mode="lines", line=dict(color=CYAN, width=2),
            name="mean |actual − expected|", hovertemplate="%{y:.1f} bytes<extra></extra>"))
        atk = np.where(pr != 0)[0]
        if len(atk):
            fig.add_trace(go.Scatter(
                x=atk, y=res_series[atk], mode="markers",
                marker=dict(color=RED, size=5), name="flagged as attack",
                hovertemplate="%{y:.1f} bytes<extra></extra>"))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", y=1.16, x=0),
                          yaxis_title="bytes", xaxis_title="frame")
        st.plotly_chart(style_fig(fig, height=232), use_container_width=True,
                        key=f"res_{shown}")

    section("Attention across ECU streams", small=True)
    if shown and ss.alpha.size:
        a = ss.alpha[-1]
        order = np.argsort(a)[::-1][:6]
        fig2 = go.Figure(go.Bar(
            x=a[order][::-1], y=[labels[i] for i in order][::-1], orientation="h",
            marker=dict(color=[BLUE if i != order[0] else CYAN for i in order][::-1]),
            hovertemplate="%{x:.3f}<extra></extra>"))
        fig2.update_layout(xaxis_title="attention weight", yaxis_title=None)
        st.plotly_chart(style_fig(fig2, height=210), use_container_width=True,
                        key=f"att_{shown}")

# ---------------------------------------------------------------- scenario facts
with st.expander("▶  What exactly is being replayed?"):
    comp = ", ".join(f"{k}: {v:,}" for k, v in sc["composition"].items())
    lr, ar = sc["lead_range"], sc["attack_range"]
    st.markdown(f"""
**Two real recordings, played back to back — not one continuous drive.**

| Segment | Frames | Source |
|---|---|---|
| Lead-in (first {lead_n}) | {lr[0]:,}–{lr[1]-1:,} | normal-driving capture |
| Attack ({n_frames - lead_n}) | {ar[0]:,}–{ar[1]:,} | attack capture |

This join is necessary and is stated plainly rather than disguised: in GEM-CAN the
attack capture is **already under attack from its first frame** — brake tampering starts
at t = 8.2 s and repeats every ~11 s — so no stretch of that recording runs clean and
then turns malicious. The lead-in supplies genuine pre-attack traffic.

Every frame is real and unmodified, and each is classified using **its own** true
preceding context, so no prediction depends on the join.

- **Composition:** {comp}
- **Held-out share:** {sc['test_share']}% of these frames are in the test split.
- **Detector:** TwinShield-Fusion — CNN-BiLSTM over the raw window plus the GRU
  Digital Twin residual, with adaptive attention across ECU streams.
- **Per tick:** {STEP} frames are revealed and a real forward pass runs on them.

The *Suspicious* (🟡) state appears when the model still predicts Normal but its
confidence has fallen below 90% — a genuine drop in certainty, not a scripted step.
""")

# ---------------------------------------------------------------- drive the loop
if ss.playing and ss.cursor < n_frames:
    time.sleep(TICK)
    st.rerun()
elif ss.playing:
    ss.playing = False
    st.rerun()
