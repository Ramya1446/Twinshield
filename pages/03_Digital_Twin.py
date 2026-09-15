"""
Page 3 -- Digital Twin.  Answers "how does your Digital Twin work?"

Every curve on this page is a stored output of the trained GRU: `expected` is the
model prediction for the next payload on that CAN ID, `actual` is the recorded
byte from the dataset, and the residual is their absolute difference.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from preprocessing.loader import CLASS_NAMES, DATA_COLS
from ui.components import flag, metric_tile, note, section
from ui.data import (data_ready, get_prepared, get_results, get_twin_outputs,
                     missing_notice, models_ready)
from ui.theme import AMBER, CYAN, GREEN, MUTED, RED, TEXT, VIOLET, style_fig

st.markdown("## 🧠 Digital Twin")
st.markdown(f'<div style="color:{MUTED};margin-top:-.5rem;font-size:1.02rem">'
            f'Learning what normal CAN behaviour should look like</div>',
            unsafe_allow_html=True)

if not data_ready():
    flag("⚠️ GEM-CAN dataset not found — see Overview.")
    st.stop()
if not models_ready():
    missing_notice()
    st.stop()

p = get_prepared()
residual, expected = get_twin_outputs()
res = get_results()
dt_metrics = res["digital_twin"]

st.write("")
st.markdown(
    '<div class="ts-card"><p style="color:#E6EDF7;font-size:.98rem;line-height:1.7">'
    'The Digital Twin is a <b>GRU-based model trained using normal CAN traffic</b>. '
    'It predicts the <b>expected next CAN payload</b> for each ECU. The difference '
    'between expected and observed behaviour is used as a <b>residual feature</b>.'
    '</p></div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- concept strip
c = st.columns([1, .12, 1, .12, 1])
concept = [("📨", "Previous CAN Messages", "the last 8 payloads on this ID"),
           ("🧠", "GRU", "learnt from normal traffic only"),
           ("🎯", "Expected Next Message", "predicted 8 payload bytes")]
for col, (ico, nm, ds) in zip([c[0], c[2], c[4]], concept):
    with col:
        st.markdown(f'<div class="ts-stage done" style="min-height:104px">'
                    f'<span class="ico">{ico}</span><span class="nm">{nm}</span>'
                    f'<span class="ds">{ds}</span></div>', unsafe_allow_html=True)
for col in (c[1], c[3]):
    with col:
        st.markdown(f'<div style="text-align:center;padding-top:2.2rem;color:{MUTED}">→</div>',
                    unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- selector
df = p.df
counts = df.groupby("arbitration_id_hex").size()
# Offer IDs that carry enough normal traffic for the twin to have learnt them.
normal_counts = df[df["y"] == 0].groupby("arbitration_id_hex").size()
eligible = sorted([i for i in counts.index if normal_counts.get(i, 0) >= 200])
# Default to an ID that carries both genuine and tampered traffic.
attacked = df[df["y"] != 0].groupby("arbitration_id_hex").size()
default = next((i for i in eligible if attacked.get(i, 0) > 0 and i != "00000000"),
               eligible[0])

s1, s2, s3 = st.columns([1.1, 1.1, 1.5])
with s1:
    can_id = st.selectbox("Select CAN ID", eligible, index=eligible.index(default))

# Default to whichever byte actually separates attack from normal on this ID --
# a measured choice, recomputed per CAN ID rather than a fixed guess.
_m = (df["arbitration_id_hex"].to_numpy() == can_id)
_n, _a = _m & (p.windows.y == 0), _m & (p.windows.y != 0)
if _a.sum() and _n.sum():
    _sep = residual[_a].mean(axis=0) - residual[_n].mean(axis=0)
    best_byte = int(np.argmax(_sep))
else:
    best_byte = int(np.argmax(residual[_m].mean(axis=0))) if _m.sum() else 0

with s2:
    byte_i = st.selectbox("Payload byte", list(range(8)), index=best_byte,
                          format_func=lambda b: f"DATA_{b}"
                          + (" ★ most separating" if b == best_byte else ""))
with s3:
    max_pts = st.slider("Frames to display", 120, 900, 340, step=20)

# Frames carrying this CAN ID, drawn from BOTH captures: the tail of the
# normal-driving recording followed by the head of the attack recording.  Taking
# a plain first-N slice would return only normal-capture frames and the chart
# would never show a deviation.
_all = np.where(df["arbitration_id_hex"].to_numpy() == can_id)[0]
_src = df["source"].to_numpy()
_norm_pool = _all[_src[_all] == "normal"]
_atk_pool = _all[_src[_all] == "attack"]

if len(_atk_pool):
    half = max_pts // 2
    sel = np.concatenate([_norm_pool[-min(half, len(_norm_pool)):],
                          _atk_pool[:max_pts - min(half, len(_norm_pool))]])
    boundary = int(min(half, len(_norm_pool)))
else:
    sel = _all[:max_pts]
    boundary = None
actual_b = df[DATA_COLS[byte_i]].to_numpy()[sel].astype(float)
expect_b = expected[sel, byte_i] * 255.0
resid_b = np.abs(actual_b - expect_b)
y_sel = p.windows.y[sel]

# ---------------------------------------------------------------- expected vs actual
section("Expected vs Actual",
        f"GRU prediction against the recorded value of DATA_{byte_i} on CAN ID {can_id}."
        + (" Frames left of the dashed line come from the normal-driving capture, "
           "frames right of it from the attack capture." if boundary else ""))

fig = go.Figure()
fig.add_trace(go.Scatter(y=expect_b, mode="lines", name="Expected",
                         line=dict(color=CYAN, width=2, dash="dot"),
                         hovertemplate="expected %{y:.1f}<extra></extra>"))
fig.add_trace(go.Scatter(y=actual_b, mode="lines", name="Actual",
                         line=dict(color=TEXT, width=1.8),
                         hovertemplate="actual %{y:.1f}<extra></extra>"))
for cls_idx, colour in ((1, RED), (2, AMBER), (3, VIOLET)):
    pts = np.where(y_sel == cls_idx)[0]
    if len(pts):
        fig.add_trace(go.Scatter(
            x=pts, y=actual_b[pts], mode="markers", name=CLASS_NAMES[cls_idx],
            marker=dict(color=colour, size=6, symbol="x"),
            hovertemplate=f"{CLASS_NAMES[cls_idx]}: %{{y:.1f}}<extra></extra>"))
if boundary:
    fig.add_vline(x=boundary, line_dash="dash", line_color=MUTED,
                  annotation_text="attack capture begins",
                  annotation_font_color=MUTED, annotation_font_size=11)
fig.update_layout(yaxis_title=f"DATA_{byte_i} value (0–255)", xaxis_title="frame",
                  legend=dict(orientation="h", y=1.14, x=0))
st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

# ---------------------------------------------------------------- residual
section("Residual", "Residual = |Actual − Expected|")

fig2 = go.Figure()
normal_pts = np.where(y_sel == 0)[0]
attack_pts = np.where(y_sel != 0)[0]
fig2.add_trace(go.Scatter(y=resid_b, mode="lines", name="Residual",
                          line=dict(color=CYAN, width=1.8),
                          hovertemplate="%{y:.1f}<extra></extra>"))
if len(attack_pts):
    fig2.add_trace(go.Scatter(x=attack_pts, y=resid_b[attack_pts], mode="markers",
                              name="attack frames", marker=dict(color=RED, size=5),
                              hovertemplate="%{y:.1f}<extra></extra>"))
if len(normal_pts):
    fig2.add_hline(y=float(resid_b[normal_pts].mean()), line_dash="dash",
                   line_color=GREEN, annotation_text="mean residual on normal frames",
                   annotation_font_color=GREEN, annotation_font_size=11)
if boundary:
    fig2.add_vline(x=boundary, line_dash="dash", line_color=MUTED)
fig2.update_layout(yaxis_title="bytes", xaxis_title="frame",
                   legend=dict(orientation="h", y=1.16, x=0))
st.plotly_chart(style_fig(fig2, height=250), use_container_width=True)

if len(attack_pts) and len(normal_pts):
    mn, ma = float(resid_b[normal_pts].mean()), float(resid_b[attack_pts].mean())
    verdict = ("higher on attack frames" if ma > mn else
               "not higher on attack frames for this byte")
    tone = GREEN if ma > mn else AMBER
    note(f'On CAN ID {can_id}, DATA_{byte_i}: mean residual is <b>{mn:.2f}</b> bytes on '
         f'normal frames and <b style="color:{tone}">{ma:.2f}</b> bytes on attack frames '
         f'— the residual is {verdict}. Values are computed from the displayed frames only.')
else:
    note("This CAN ID carries no attack frames in the selected range, so only normal "
         "behaviour is shown.")

# Whole-ID view across all 8 bytes, so the byte selector cannot flatter the result.
if _a.sum() and _n.sum():
    st.markdown("**Residual per payload byte — every byte on this CAN ID**")
    mn_all = residual[_n].mean(axis=0) * 255
    ma_all = residual[_a].mean(axis=0) * 255
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Normal frames", x=[f"DATA_{i}" for i in range(8)],
                          y=mn_all, marker_color=GREEN,
                          hovertemplate="%{y:.2f} bytes<extra>Normal</extra>"))
    fig3.add_trace(go.Bar(name="Attack frames", x=[f"DATA_{i}" for i in range(8)],
                          y=ma_all, marker_color=RED,
                          hovertemplate="%{y:.2f} bytes<extra>Attack</extra>"))
    fig3.update_layout(barmode="group", yaxis_title="mean |residual| (bytes)",
                       legend=dict(orientation="h", y=1.2, x=0))
    st.plotly_chart(style_fig(fig3, height=250), use_container_width=True)

    overall_n, overall_a = float(mn_all.mean()), float(ma_all.mean())
    if overall_a > overall_n * 1.5:
        note(f"Across all bytes on {can_id} the mean residual rises from "
             f"<b>{overall_n:.2f}</b> to <b style='color:{GREEN}'>{overall_a:.2f}</b> "
             f"bytes under attack — the Digital Twin separates this attack clearly.")
    else:
        note(f"Across all bytes on {can_id} the mean residual is <b>{overall_n:.2f}</b> "
             f"on normal frames and <b style='color:{AMBER}'>{overall_a:.2f}</b> under "
             f"attack. <b>The residual does not separate this attack on this ID</b>, and "
             f"that is reported rather than hidden — detection of this class has to come "
             f"from the timing and sequence features the detector also sees, not from the "
             f"Digital Twin alone.")

st.write("")

# ---------------------------------------------------------------- metrics
section("Digital Twin prediction quality",
        "Measured on held-out <b>normal</b> frames only — the twin models normal "
        "behaviour, so this is where its accuracy is meaningful.")
m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_tile("MAE", f"{dt_metrics['MAE']:.2f}", "bytes (0–255 scale)", highlight=True)
with m2:
    metric_tile("RMSE", f"{dt_metrics['RMSE']:.2f}", "bytes")
with m3:
    metric_tile("MSE", f"{dt_metrics['MSE']:.1f}", "bytes²")
with m4:
    metric_tile("Eval frames", f"{dt_metrics.get('n_val_normal_frames', 0):,}",
                "held-out normal frames")

# ---------------------------------------------------------------- technical
st.write("")
with st.expander("▶  How does the Digital Twin work?"):
    st.markdown(r"""
**In plain English.** Each ECU on the CAN bus repeats a fairly predictable pattern of
messages. The Digital Twin watches only *clean* traffic and learns that pattern, so it
can say what the next message on a given CAN ID *should* look like. When an attacker
injects a forged frame, the real bytes stop matching the learnt pattern and the gap
between them grows. That gap is the residual.

**Architecture.**

| Component | Setting |
|---|---|
| Input | last 8 payloads observed on the same CAN ID, scaled to `[0,1]` |
| ID embedding | 16-dim, so one network serves every ECU |
| Recurrent core | GRU, 96 hidden units |
| Head | 64-unit ReLU layer → 8 outputs → sigmoid |
| Loss | MSE against the true next payload |
| Training data | **Normal frames from the training split only** |

**Formula.**
""")
    st.latex(r"\text{Residual}_t \;=\; \bigl| \text{Actual}_t - \text{Predicted}_t \bigr|")
    st.markdown(f"""
computed per payload byte, giving an 8-dimensional residual vector for every frame.
Those vectors feed the detector, and their per-ECU averages feed the adaptive fusion
attention.

**Why train on normal traffic only?** The twin must not learn what attacks look like —
otherwise it could only catch attacks it had already seen. Because it models normal
behaviour, an unusual payload raises the residual whether or not that attack was in
the training data.

**Honest reading of MAE = {dt_metrics['MAE']:.2f} bytes.** CAN payloads carry several
independent signals, and some bytes (counters, noisy sensor readings) are inherently
hard to predict. What matters for detection is not that the residual is near zero,
but that it is *reliably different* between normal and tampered traffic on the same
CAN ID — which the chart above lets you check directly.
""")
