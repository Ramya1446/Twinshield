"""Page 1 -- Overview. The judge should understand the project in 10-15 seconds."""
from __future__ import annotations

import streamlit as st

from ui.components import card, flag, kv_card, note, pipeline, section
from ui.data import (data_ready, get_prepared, get_results, get_zero_day,
                     models_ready, zeroday_ready)
from ui.theme import CYAN, MUTED, RED

# ---------------------------------------------------------------- hero
st.markdown(
    '<div class="ts-hero">'
    '<span class="ts-eyebrow">Automotive Intrusion Detection · Real Vehicle Data</span>'
    '<h1>TwinShield-Fusion</h1>'
    '<div class="sub">Catching CAN-bus attacks nobody trained the model on</div>'
    '<div class="desc">A Digital Twin learns what normal vehicle communication looks '
    'like. Anything that deviates raises a residual — so an attack can be flagged '
    'because it is <i>abnormal</i>, not because it matches a signature somebody '
    'already collected and labelled.</div>'
    '</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- pipeline
section("How it works",
        "Every stage below runs on real CAN frames — nothing in this application is "
        "simulated.")
pipeline()
note("Two of these stages carry the decision. The <b>CNN-BiLSTM</b> names attacks it "
     "was trained on; the <b>Anomaly Gate</b> reads the Digital Twin residual and can "
     "raise an alert on traffic that matches no label it has ever seen. Only the "
     "second one survives an attack that was not in the training set.")

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
        card("🧠", "Digital Twin",
             "A GRU trained on normal traffic only. It never sees an attack, which is "
             "why an unseen one still raises its residual.")
        st.write("")
        card("🔗", "Adaptive Fusion",
             "Attention weighs residual evidence across per-ECU streams, so evidence "
             "concentrates on whichever ECU is under attack.")
    with c2:
        card("🛡️", "Unknown-Attack Gate",
             "Thresholds the residual against normal behaviour alone. Calibrated "
             "without a single attack frame.")
        st.write("")
        card("🔍", "Explainable AI",
             "SHAP attributions and live attention weights show which feature and "
             "which ECU drove each decision.")

st.write("")

# ---------------------------------------------------------------- headline result
if zeroday_ready():
    zd = get_zero_day()
    zdm = zd["models"]
    unknown = zd["unknown_class"]
    if "TwinShield-Fusion" in zdm:
        baselines = {m: zdm[m]["unknown_detection_rate"] for m in zdm
                     if m != "TwinShield-Fusion"}
        prop = zdm["TwinShield-Fusion"]["unknown_detection_rate"]
        best_base = max(baselines.values()) if baselines else 0.0
        section("The headline result")
        cols = st.columns(len(baselines) + 1)
        for col, (m, v) in zip(cols, baselines.items()):
            with col:
                st.markdown(
                    f'<div class="ts-metric"><div class="lbl">{m}</div>'
                    f'<div class="val" style="color:{RED if v < 0.05 else MUTED}">'
                    f'{v*100:.0f}%</div>'
                    f'<div class="sub">of held-out {unknown} frames flagged</div></div>',
                    unsafe_allow_html=True)
        with cols[-1]:
            st.markdown(
                f'<div class="ts-metric hi"><div class="lbl">TwinShield-Fusion</div>'
                f'<div class="val">{prop*100:.0f}%</div>'
                f'<div class="sub">of held-out {unknown} frames flagged</div></div>',
                unsafe_allow_html=True)
        note(f"<b>{unknown}</b> was removed from every training stage — from the label "
             f"space, from the context windows, from the Digital Twin's history and "
             f"from the gate's calibration set — then all four models were asked about "
             f"a test split that still contains "
             f"{zd['test_class_support'].get(unknown, 0):,} of its frames. "
             f"TwinShield-Fusion raises "
             f"{zdm['TwinShield-Fusion']['false_positive_rate']*100:.2f}% false alarms "
             f"on normal traffic to get there. Watch it happen frame by frame on the "
             f"Live Monitor; the full experiment is on Results.")
        st.write("")

if models_ready():
    res = get_results()
    variant = "with_id" if "with_id" in res["results"] else next(iter(res["results"]))
    r = res["results"][variant]
    if "TwinShield-Fusion" in r:
        f1 = r["TwinShield-Fusion"]["metrics"]["Macro F1"]
        base = r.get("CNN-BiLSTM", {}).get("metrics", {}).get("Macro F1")
        sub = f"CNN-BiLSTM baseline: {base:.4f}" if base is not None else ""
        st.markdown(
            f'<div class="ts-card" style="border-color:rgba(169,130,47,.35)">'
            f'<h4>Closed-set control — every attack class in training</h4>'
            f'<p>TwinShield-Fusion macro-F1 <b style="color:{CYAN};font-size:1.25rem">'
            f'{f1:.4f}</b> &nbsp;·&nbsp; {sub}. On this conventional benchmark every '
            f'model scores about the same, which is the point: a closed-set score says '
            f'how well a model handles attacks somebody already labelled.</p></div>',
            unsafe_allow_html=True)
        st.write("")

# ---------------------------------------------------------------- CTA
c1, c2, c3 = st.columns([1, 1.1, 1])
with c2:
    if st.button("▶  START LIVE DEMO", type="primary", use_container_width=True):
        st.switch_page("pages/02_Live_Monitor.py")

missing = []
if not models_ready():
    missing.append("<code>python train.py</code> (closed-set comparison)")
if not zeroday_ready():
    missing.append("<code>python train_zeroday.py</code> (unknown-attack experiment)")
if missing:
    st.write("")
    flag("⚠️ <b>Not all artifacts are present.</b> Run " + " and ".join(missing)
         + " once before the presentation so the demo loads instantly.")

# ---------------------------------------------------------------- technical
st.write("")
with st.expander("▶  Technical details"):
    st.markdown(f"""
**Problem.** Controller Area Network traffic has no authentication, so any node that
reaches the bus can inject frames. Published CAN intrusion detectors are trained on a
fixed catalogue of attacks — and a vehicle on the road meets attacks that were not in
anybody's catalogue.

**Approach.**
1. A **GRU Digital Twin** is trained on *normal* traffic only. Given the previous
   payloads seen on a CAN ID, it predicts the next payload for that ID.
2. The **residual** `|actual − predicted|` measures how far observed traffic has
   drifted from learnt-normal behaviour. It is defined for any frame whatsoever,
   including one belonging to an attack nobody has labelled.
3. **Adaptive multi-ECU fusion** applies attention across per-CAN-ID residual
   streams, so evidence concentrates on whichever ECU is being attacked.
4. A **CNN-BiLSTM** classifies the frame from the raw window plus the fused residual.
5. An **anomaly gate**, calibrated on normal frames alone, raises an alert when the
   residual deviates further than normal traffic ever does — the channel that does
   not depend on a known attack.
6. **SHAP** attributes the decision back to individual features.

**Classification granularity.** Each frame is classified using the
{p.windows.gather_idx.shape[1]} frames of context that precede it. Frame-level rather
than window-level, because brake and steering tampering are injected as an interleaved
pair and would otherwise share a label.

**Dataset.** GEM-CAN (Zenodo 17834776), CC-BY-NC-3.0, {total:,} real frames from a
GEM e6 autonomous electric vehicle on a closed track.

**Three pages.** *Overview* (here) · *Live Monitor* — all four models racing on the
same two replay cases · *Results* — the full unknown-attack experiment, the Digital
Twin, the closed-set comparison, SHAP and the dataset audit.
""")
