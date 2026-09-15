"""
Page 4 -- Results.  Answers "why should we believe your method?"

Every number is read from results/results.json, which is written by train.py from
predictions on one held-out test split shared by all models.  The closing research
statement is generated from the measured difference and is not allowed to claim an
improvement that the numbers do not show.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from explainability.shap_explainer import aggregate, load as load_shap
from preprocessing.loader import CLASS_NAMES
from ui.components import flag, metric_tile, note, section
from ui.data import data_ready, get_audit, get_results, missing_notice, models_ready
from ui.theme import (AMBER, BLUE, CYAN, GREEN, MUTED, RED, TEXT, VIOLET,
                      CLASS_COLORS, style_fig)

st.markdown("## 📊 Model Performance")

if not data_ready():
    flag("⚠️ GEM-CAN dataset not found — see Overview.")
    st.stop()
if not models_ready():
    missing_notice()
    st.stop()

res = get_results()
variants = list(res["results"].keys())
labels = {"with_id": "CAN ID included", "no_id": "CAN ID excluded (controlled)"}

vsel = st.radio("Feature set", variants, horizontal=True,
                format_func=lambda v: labels.get(v, v))
R = res["results"][vsel]
order = [m for m in res["model_order"] if m in R]
support = res["test_class_support"]

note(f"All models evaluated on the same held-out test split "
     f"({res['split_sizes']['test']:,} frames). Class support — "
     + " · ".join(f"{k}: {v:,}" for k, v in support.items()))

if vsel == "no_id":
    note("In this variant the arbitration ID is removed from the model inputs. The "
         "fusion branch additionally <b>rank-orders</b> its ECU slots by residual "
         "magnitude instead of keeping one fixed slot per CAN ID — otherwise "
         "“slot 1 is active” would be equivalent to “ID 00000000 is present”, "
         "handing the shortcut straight back to the model.")

st.write("")

# ---------------------------------------------------------------- model cards
cards = [m for m in ["CNN", "BiLSTM", "CNN-BiLSTM", "TwinShield-Fusion"] if m in R]
cols = st.columns(len(cards))
best = max(R[m]["metrics"]["Macro F1"] for m in order)
for col, name in zip(cols, cards):
    f1 = R[name]["metrics"]["Macro F1"]
    with col:
        metric_tile(name, f"{f1:.4f}", "Macro F1",
                    highlight=abs(f1 - best) < 1e-9,
                    colour=CYAN if abs(f1 - best) < 1e-9 else None)

st.write("")

# ---------------------------------------------------------------- comparison table
section("Model comparison", "Computed from actual predictions on the shared test set.")
rows = []
for m in order:
    mm = R[m]["metrics"]
    rows.append({
        "Model": m,
        "Accuracy": round(mm["Accuracy"], 4),
        "Precision": round(mm["Precision (macro)"], 4),
        "Recall": round(mm["Recall (macro)"], 4),
        "Macro F1": round(mm["Macro F1"], 4),
        "Weighted F1": round(mm["Weighted F1"], 4),
    })
table = pd.DataFrame(rows)


def _shade(col: pd.Series) -> list:
    """Blue gradient over a column without pulling in matplotlib."""
    lo, hi = float(col.min()), float(col.max())
    span = hi - lo
    out = []
    for v in col:
        t = 0.0 if span <= 0 else (float(v) - lo) / span
        out.append(f"background-color: rgba(59,130,246,{0.08 + 0.42 * t:.3f})")
    return out


st.dataframe(
    table.style.apply(_shade, subset=["Macro F1"]).format(precision=4),
    use_container_width=True, hide_index=True)

# If every model lands in the same tiny band, say so -- a 0.001 gap between
# architectures is not evidence that one is better.
spread = table["Macro F1"].max() - table["Macro F1"].min()
if spread < 0.01:
    note(f"⚠️ <b>All models fall within {spread:.4f} macro-F1 of each other.</b> On this "
         f"feature set the task is effectively <b>saturated</b>, so the differences "
         f"between architectures are not meaningful and should not be read as one "
         f"method beating another."
         + (" This is expected here: arbitration ID alone already separates DoS "
            "perfectly, so most of the label information is available before any "
            "sequence model runs." if vsel == "with_id" else ""))

# ---------------------------------------------------------------- per-class F1
section("Per-class F1", "Where the difficulty actually lies.")
pc = pd.DataFrame({m: R[m]["per_class_f1"] for m in order}).T
pc.index.name = "Model"

d1, d2, _ = st.columns([1, 1, 2])
with d1:
    st.download_button("⤓  Comparison table (CSV)", table.to_csv(index=False),
                       file_name=f"twinshield_comparison_{vsel}.csv",
                       mime="text/csv", use_container_width=True)
with d2:
    st.download_button("⤓  Per-class F1 (CSV)", pc.round(4).to_csv(),
                       file_name=f"twinshield_per_class_f1_{vsel}.csv",
                       mime="text/csv", use_container_width=True)
fig = go.Figure()
for cls in CLASS_NAMES:
    if cls in pc.columns:
        fig.add_trace(go.Bar(name=cls, x=pc.index, y=pc[cls],
                             marker_color=CLASS_COLORS[cls],
                             hovertemplate="%{y:.4f}<extra>" + cls + "</extra>"))
fig.update_layout(barmode="group", yaxis_title="F1", yaxis_range=[0, 1.05],
                  legend=dict(orientation="h", y=1.15, x=0))
st.plotly_chart(style_fig(fig, height=330), use_container_width=True)

# ---------------------------------------------------------------- ablation
section("Ablation — the research contribution",
        "Each step adds one component to the same CNN-BiLSTM backbone.")
abl = [m for m in res["ablation_order"] if m in R]
vals = [R[m]["metrics"]["Macro F1"] for m in abl]

cols = st.columns(len(abl) * 2 - 1)
for i, (name, v) in enumerate(zip(abl, vals)):
    with cols[i * 2]:
        delta = v - vals[0]
        sub = "baseline" if i == 0 else f"{delta:+.4f} vs baseline"
        metric_tile(name, f"{v:.4f}", sub, highlight=(i == len(abl) - 1))
    if i < len(abl) - 1:
        with cols[i * 2 + 1]:
            st.markdown(f'<div style="text-align:center;padding-top:2.4rem;'
                        f'color:{MUTED};font-size:1.3rem">→</div>',
                        unsafe_allow_html=True)

fig2 = go.Figure(go.Scatter(
    x=abl, y=vals, mode="lines+markers+text",
    text=[f"{v:.4f}" for v in vals], textposition="top center",
    textfont=dict(color=TEXT, size=12),
    line=dict(color=CYAN, width=3), marker=dict(size=11, color=BLUE)))
lo, hi = min(vals), max(vals)
pad = max((hi - lo) * 0.35, 0.004)
fig2.update_layout(yaxis_title="Macro F1", yaxis_range=[lo - pad, hi + pad * 1.6])
st.plotly_chart(style_fig(fig2, height=290), use_container_width=True)

# ---------------------------------------------------------------- SHAP
st.markdown("## 🔍 Why did the model detect this attack?")
shap_payload = load_shap()
if shap_payload is None:
    flag("SHAP values not computed yet. Run <code>python compute_shap.py</code> "
         "to generate them.")
else:
    scls = st.selectbox(
        "Explain predictions of class",
        ["All predictions"] + [c for c in CLASS_NAMES],
        index=1 if len(CLASS_NAMES) > 1 else 0)
    cidx = None if scls == "All predictions" else CLASS_NAMES.index(scls)
    names, imp = aggregate(shap_payload, cidx, top_k=12)

    if len(names) == 0:
        note(f"The model made no predictions of class “{scls}” in the explained sample.")
    else:
        c1, c2 = st.columns([1.45, 1], gap="large")
        with c1:
            figs = go.Figure(go.Bar(
                x=imp[::-1], y=names[::-1], orientation="h",
                marker=dict(color=[CYAN if i == 0 else BLUE
                                   for i in range(len(names))][::-1]),
                hovertemplate="%{x:.5f}<extra></extra>"))
            figs.update_layout(xaxis_title="mean |SHAP value|", yaxis_title=None)
            st.plotly_chart(style_fig(figs, height=380), use_container_width=True)
        with c2:
            ecu = shap_payload["ecu_shap"]
            slabels = list(shap_payload["stream_labels"])
            if cidx is not None:
                sel = shap_payload["pred"] == cidx
                ecu = ecu[sel] if sel.sum() else ecu
            eimp = np.abs(ecu).mean(axis=0)
            eorder = np.argsort(eimp)[::-1][:8]
            fige = go.Figure(go.Bar(
                x=eimp[eorder][::-1], y=[slabels[i] for i in eorder][::-1],
                orientation="h", marker=dict(color=VIOLET),
                hovertemplate="%{x:.5f}<extra></extra>"))
            fige.update_layout(xaxis_title="mean |SHAP| per ECU stream", yaxis_title=None)
            st.plotly_chart(style_fig(fige, height=380, title="Which ECU stream mattered"),
                            use_container_width=True)

        note("Features with larger positive SHAP values contributed more strongly "
             "toward the selected prediction. Attributions are summed over the "
             f"{res['config']['window']}-frame context window and averaged over "
             f"{len(shap_payload['pred']):,} explained test frames.")

# ---------------------------------------------------------------- research validity
st.write("")
section("Research validity")
audit = get_audit()
sc = audit["can_id_shortcut"]
dup = audit["duplicates"]

a1, a2, a3 = st.columns(3)
with a1:
    metric_tile("CAN ID alone — accuracy", f"{sc['accuracy']*100:.2f}%",
                "majority-class lookup on ID", colour=RED)
with a2:
    metric_tile("CAN ID alone — macro F1", f"{sc['macro_f1']:.4f}",
                "same lookup, macro-averaged", colour=AMBER)
with a3:
    metric_tile("Duplicate frames", f"{dup['duplicate_share_pct']:.1f}%",
                f"{dup['test_frames_with_identical_twin_in_train_pct']:.1f}% of test "
                f"frames have a twin in train", colour=AMBER)

per = sc["per_class_f1"]
flag(
    "<b>Reported openly:</b> a lookup table using nothing but the arbitration ID "
    f"reaches <b>{sc['accuracy']*100:.2f}% accuracy</b> on this dataset — because every "
    "DoS frame carries ID <code>00000000</code> and no normal frame does "
    f"(DoS F1 = {per['DoS']:.2f}). The same lookup scores "
    f"<b>{per['Brake Tampering']:.2f}</b> and <b>{per['Steering Tampering']:.2f}</b> on "
    "brake and steering tampering, which reuse legitimate IDs. "
    "<br><br>Two consequences, and they shape how this project should be read: "
    "<b>accuracy is not a meaningful headline metric here</b>, and the value of the "
    "Digital Twin residual has to be judged on the tampering classes. The "
    "<i>CAN ID excluded</i> feature set above re-runs every model without arbitration "
    "ID so the shortcut is removed entirely.")

with st.expander("▶  Full dataset audit"):
    st.markdown("**Class balance**")
    st.dataframe(pd.DataFrame(audit["class_balance"]), use_container_width=True,
                 hide_index=True)
    st.markdown("**Duplicate analysis**")
    st.json(dup)
    st.markdown("**CAN ID usage — normal vs attack**")
    ov = pd.DataFrame(audit["can_id_overlap"])
    st.dataframe(ov, use_container_width=True, hide_index=True)
    st.markdown(f"""
**Splitting.** Frames are cut into contiguous blocks of 256 and whole blocks are
assigned to train/val/test, stratified by the rarest class each block contains. A
frame's context window never leaves its block, so no frame is shared between splits.
A purely chronological split was rejected because the tampering attacks occupy the
opening ~9% of the attack capture and would leave the test set with none.

**Remaining limitation.** Blocks adjacent in time can still be correlated, and CAN
traffic is highly repetitive ({dup['duplicate_share_pct']:.1f}% of frames are exact
duplicates of another frame). Results should therefore be read as within-capture
performance, not as evidence of generalisation to a different vehicle or drive cycle.
""")

# ---------------------------------------------------------------- statement
st.write("")
section("Result statement")

base_name = "CNN-BiLSTM"
prop_name = "TwinShield-Fusion"
if base_name in R and prop_name in R:
    b = R[base_name]["metrics"]["Macro F1"]
    t = R[prop_name]["metrics"]["Macro F1"]
    delta_pp = (t - b) * 100

    if delta_pp > 0.5:
        msg = (f"TwinShield-Fusion achieved a higher Macro-F1 than the CNN-BiLSTM "
               f"baseline by <b>{delta_pp:.2f} percentage points</b> "
               f"({b:.4f} → {t:.4f}) on the {labels.get(vsel, vsel)} feature set.")
        tone = GREEN
    elif delta_pp < -0.5:
        msg = (f"TwinShield-Fusion did <b>not</b> improve classification performance in "
               f"this experiment ({b:.4f} → {t:.4f}, {delta_pp:.2f} percentage points), "
               f"highlighting the limitations of the current residual representation.")
        tone = RED
    else:
        msg = (f"TwinShield-Fusion achieved <b>comparable</b> classification performance "
               f"({b:.4f} → {t:.4f}, {delta_pp:+.2f} percentage points) while adding "
               f"Digital Twin residual analysis, adaptive fusion, and explainability.")
        tone = AMBER

    tamper = [c for c in ("Brake Tampering", "Steering Tampering") if c in R[prop_name]["per_class_f1"]]
    if tamper:
        bt = np.mean([R[base_name]["per_class_f1"][c] for c in tamper])
        tt = np.mean([R[prop_name]["per_class_f1"][c] for c in tamper])
        msg += (f"<br><br>On the two tampering classes — the ones arbitration ID cannot "
                f"separate — mean F1 moved from <b>{bt:.4f}</b> to <b>{tt:.4f}</b> "
                f"({(tt-bt)*100:+.2f} pp).")

    st.markdown(
        f'<div class="ts-card" style="border-color:{tone}55;background:{tone}10">'
        f'<p style="color:{TEXT};font-size:1.0rem;line-height:1.75">{msg}</p></div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------- technical
st.write("")
with st.expander("▶  Model management (train / save / load)"):
    import os as _os
    import time as _time

    SAVED_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "saved_models")

    st.markdown("""
Models are trained **offline, before the presentation**, so nothing here is required
during a demo. Training from inside the app is deliberately not offered — it would risk
a multi-minute stall in front of an audience.

```bash
python train.py                      # train everything, save to saved_models/
python train.py --epochs 3           # quick rehearsal run
python train.py --variants no_id     # only the ID-free controlled experiment
python train.py --only twinshield    # retrain one architecture, merge the rest
python compute_shap.py               # SHAP attributions
python check_demo.py                 # verify every artifact before presenting
```

**Saving** is automatic: checkpoints are written as
`{variant}__{architecture}__{feature_mode}.pt`. **Loading** happens on first page use
and is cached for the session.
""")

    if _os.path.isdir(SAVED_DIR):
        files = sorted(f for f in _os.listdir(SAVED_DIR) if f.endswith((".pt", ".npy")))
        st.dataframe(pd.DataFrame([{
            "Artifact": f,
            "Size (KB)": round(_os.path.getsize(_os.path.join(SAVED_DIR, f)) / 1024, 1),
            "Saved": _time.strftime("%Y-%m-%d %H:%M",
                                    _time.localtime(_os.path.getmtime(
                                        _os.path.join(SAVED_DIR, f)))),
        } for f in files]), use_container_width=True, hide_index=True)

    if st.button("↻  Reload models from disk", key="reload_models"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Caches cleared — models and results will be re-read on next render.")

st.write("")
with st.expander("▶  Technical details"):
    cfg = res["config"]
    st.markdown(f"""
| Setting | Value |
|---|---|
| Context window | {cfg['window']} frames |
| Detector epochs | {cfg['epochs']} |
| Digital Twin epochs | {cfg['dt_epochs']} |
| Batch size | {cfg['batch']} |
| Seed | {cfg['seed']} |
| ECU streams | {cfg['n_streams']} |
| Train / Val / Test | {res['split_sizes']['train']:,} / {res['split_sizes']['val']:,} / {res['split_sizes']['test']:,} |

Model selection used the best validation macro-F1 across epochs; the selected weights
were then evaluated once on the test split. Class-weighted cross-entropy compensates
for the 152:1 imbalance between Normal and the tampering classes.

**Per-model training time (s)**
""")
    st.dataframe(pd.DataFrame([
        {"Model": m, "Features": R[m]["n_features"],
         "Best val macro F1": round(R[m]["best_val_macro_f1"], 4),
         "Train seconds": R[m]["train_seconds"]} for m in order
    ]), use_container_width=True, hide_index=True)
