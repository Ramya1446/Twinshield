"""Page 1 -- Overview. The judge should understand the project in 10-15 seconds."""
from __future__ import annotations

import streamlit as st

from ui.components import card, flag, kv_card, note, pipeline, section
from ui.data import data_ready, get_prepared, get_results, models_ready
from ui.theme import CYAN

# ---------------------------------------------------------------- hero
st.markdown(
    '<div class="ts-hero">'
    '<span class="ts-eyebrow">Automotive Intrusion Detection · Real Vehicle Data</span>'
    '<h1>TwinShield-Fusion</h1>'
    '<div class="sub">AI-Powered CAN Intrusion Detection for Electric Vehicles</div>'
    '<div class="desc">Learn normal vehicle communication, detect deviations, '
    'identify attacks, and explain why they were detected.</div>'
    '</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- pipeline
section("How it works",
        "Every stage below runs on real CAN frames — nothing in this application is simulated.")
pipeline()

st.write("")

# ---------------------------------------------------------------- dataset + features
if not data_ready():
    flag("⚠️ <b>GEM-CAN dataset not found.</b> Download "
         "<code>GEM_CAN_Dataset.zip</code> from "
         "<a href='https://zenodo.org/records/17834776'>zenodo.org/records/17834776</a> "
         "and extract it into <code>data/</code>.")
    st.stop()

p = get_prepared()
dist = p.df["class_name"].value_counts()
total = int(dist.sum())
n_ids = p.df["arbitration_id"].nunique()

left, right = st.columns([1.05, 1], gap="large")

with left:
    section("Dataset")
    attack_rows = [(k, f"{int(v):,} frames") for k, v in dist.items() if k != "Normal"]
    kv_card("GEM-CAN", [
        ("Vehicle", "GEM e6 Electric Vehicle"),
        ("Data", "Real CAN traffic (250 kbps)"),
        ("Total frames", f"{total:,}"),
        ("Normal frames", f"{int(dist.get('Normal', 0)):,}"),
        ("Distinct CAN IDs", str(n_ids)),
        *attack_rows,
    ], icon="🗂️")
    note("Attack types listed are exactly those present in the dataset — "
         "DoS plus brake and steering tampering. No other classes are claimed.")

with right:
    section("What the system does")
    c1, c2 = st.columns(2)
    with c1:
        card("🧠", "Digital Twin", "Learns what normal CAN behaviour looks like.")
        st.write("")
        card("🔍", "Explainable AI", "Shows which features influenced the prediction.")
    with c2:
        card("🚨", "Attack Detection", "Identifies abnormal CAN traffic.")
        st.write("")
        card("🔗", "Adaptive Fusion", "Weighs residual evidence across ECU streams.")

st.write("")

# ---------------------------------------------------------------- headline result
if models_ready():
    res = get_results()
    variant = "with_id" if "with_id" in res["results"] else next(iter(res["results"]))
    r = res["results"][variant]
    if "TwinShield-Fusion" in r:
        f1 = r["TwinShield-Fusion"]["metrics"]["Macro F1"]
        base = r.get("CNN-BiLSTM", {}).get("metrics", {}).get("Macro F1")
        sub = f"CNN-BiLSTM baseline: {base:.4f}" if base is not None else ""
        st.markdown(
            f'<div class="ts-card" style="border-color:rgba(34,211,238,.35)">'
            f'<h4>Measured on the held-out test split</h4>'
            f'<p>TwinShield-Fusion macro-F1 <b style="color:{CYAN};font-size:1.25rem">'
            f'{f1:.4f}</b> &nbsp;·&nbsp; {sub}</p></div>', unsafe_allow_html=True)
        st.write("")

# ---------------------------------------------------------------- CTA
c1, c2, c3 = st.columns([1, 1.1, 1])
with c2:
    if st.button("▶  START LIVE DEMO", type="primary", use_container_width=True):
        st.switch_page("pages/02_Live_Monitor.py")

if not models_ready():
    st.write("")
    flag("⚠️ <b>Models not trained yet.</b> Run <code>python train.py</code> once "
         "before the presentation so the demo loads instantly.")

# ---------------------------------------------------------------- technical
st.write("")
with st.expander("▶  Technical details"):
    st.markdown(f"""
**Problem.** Controller Area Network traffic has no authentication, so any node that
reaches the bus can inject frames. This system detects such injections from traffic alone.

**Approach.**
1. A **GRU Digital Twin** is trained on *normal* traffic only. Given the previous
   payloads seen on a CAN ID, it predicts the next payload for that ID.
2. The **residual** `|actual − predicted|` measures how far observed traffic has
   drifted from learnt-normal behaviour.
3. **Adaptive multi-ECU fusion** applies attention across per-CAN-ID residual
   streams, so evidence concentrates on whichever ECU is being attacked.
4. A **CNN-BiLSTM** classifies the frame from the raw window plus the fused residual.
5. **SHAP** attributes the decision back to individual features.

**Classification granularity.** Each frame is classified using the {p.windows.gather_idx.shape[1]}
frames of context that precede it. Frame-level rather than window-level, because brake and
steering tampering are injected as an interleaved pair and would otherwise share a label.

**Dataset.** GEM-CAN (Zenodo 17834776), CC-BY-NC-3.0, {total:,} real frames from a
GEM e6 autonomous electric vehicle on a closed track.
""")
