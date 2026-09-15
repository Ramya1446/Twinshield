"""
Page 3 -- Results.  Answers "why should we believe your method?"

Three bodies of evidence, in the order a reviewer needs them:

  1. **Unknown-attack generalisation** (`results/zero_day.json`) -- the
     leave-one-attack-out experiment behind the Live Monitor, reported per model
     and per class, with the anomaly gate characterised over its whole operating
     range rather than at one flattering threshold.
  2. **The Digital Twin itself** -- expected vs actual payload, the residual it
     produces, and where that residual does and does not separate an attack.
  3. **Closed-set comparison** (`results/results.json`) -- every model on one
     shared held-out split, the ablation ladder, SHAP, and the dataset audit.

Every number is read from a file written by a training run. The closing research
statement is generated from the measured difference and is not allowed to claim
an improvement the numbers do not show.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from explainability.shap_explainer import aggregate, load as load_shap
from preprocessing.loader import CLASS_NAMES, DATA_COLS
from ui.components import flag, metric_tile, note, section
from ui.data import (data_ready, get_audit, get_prepared, get_results,
                     get_twin_outputs, get_zd_twin_outputs, get_zero_day,
                     missing_notice, models_ready, zeroday_ready)
from ui.theme import (AMBER, BLUE, CLASS_COLORS, CYAN, GREEN, MUTED, RED, TEXT,
                      VIOLET, style_fig)

st.markdown("## 📊 Results")

if not data_ready():
    flag("⚠️ GEM-CAN dataset not found — see Overview.")
    st.stop()
if not models_ready():
    missing_notice()
    st.stop()

p = get_prepared()
res = get_results()

# ===========================================================================
# 1. Unknown-attack generalisation -- the research contribution
# ===========================================================================
st.markdown("### 🛡️ Unknown-attack generalisation")

if not zeroday_ready():
    flag("⚠️ <b>Zero-day experiment not run yet.</b> Run "
         "<code>python train_zeroday.py</code> to produce this section and the "
         "Live Monitor comparison.")
else:
    zd = get_zero_day()
    unknown = zd["unknown_class"]
    zd_models = zd["models"]
    order = [m for m in zd["model_order"] if m in zd_models]
    gate = zd["gate"]
    support = zd["test_class_support"]

    note(f"One attack class — <b>{unknown}</b> — was removed from every training "
         f"stage: from the label space, from the context windows "
         f"({zd['split_sizes']['dropped_contaminated_train']:,} train and "
         f"{zd['split_sizes']['dropped_contaminated_val']:,} val frames dropped because "
         f"a window contained one), from the Digital Twin's per-ECU history, and from "
         f"the anomaly gate's calibration set. The test split was left untouched and "
         f"still holds <b>{support.get(unknown, 0):,}</b> {unknown} frames. All four "
         f"models are then asked the only question comparable across different label "
         f"spaces: <b>did you raise an alarm?</b>")

    cols = st.columns(len(order))
    best_unknown = max(zd_models[m]["unknown_detection_rate"] for m in order)
    for col, m in zip(cols, order):
        d = zd_models[m]["unknown_detection_rate"]
        with col:
            metric_tile(m, f"{d*100:.1f}%", f"of {unknown} frames flagged",
                        highlight=abs(d - best_unknown) < 1e-9,
                        colour=None if abs(d - best_unknown) < 1e-9 else
                        (RED if d == 0 else AMBER))

    st.write("")

    # ---------------------------------------------------------------- per class
    section("Alarm rate per attack class",
            "Fraction of that class's held-out test frames on which the model raised "
            "any alert. The quarantined class is marked ★.")
    attack_classes = [c for c in CLASS_NAMES[1:] if support.get(c, 0) > 0]
    fig = go.Figure()
    for c in attack_classes:
        label = f"{c} ★ unknown" if c == unknown else c
        fig.add_trace(go.Bar(
            name=label, x=order,
            y=[zd_models[m]["per_class"].get(c, {}).get("alarm_rate", 0.0)
               for m in order],
            marker_color=CLASS_COLORS.get(c, BLUE),
            marker_pattern_shape="/" if c == unknown else "",
            hovertemplate="%{y:.1%}<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group", yaxis_title="alarm rate", yaxis_range=[0, 1.06],
                      yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=1.16, x=0))
    st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    # ---------------------------------------------------------------- cost side
    section("What the extra channel costs",
            "An always-on alarm would score 100% on every attack, so the false-positive "
            "rate on normal traffic is reported next to it.")
    cost = pd.DataFrame([{
        "Model": m,
        f"{unknown} detected": f"{zd_models[m]['unknown_detection_rate']*100:.1f}%",
        "False alarms on normal": f"{zd_models[m]['false_positive_rate']*100:.2f}%",
        "Known-class macro F1": round(zd_models[m]["known_class_metrics"]["Macro F1"], 4),
        "From supervised head":
            f"{zd_models[m]['per_class'].get(unknown, {}).get('supervised_alarm_rate', 0)*100:.1f}%",
        "From anomaly gate":
            f"{zd_models[m]['per_class'].get(unknown, {}).get('gate_alarm_rate', 0)*100:.1f}%"
            if zd_models[m]["uses_anomaly_gate"] else "—",
    } for m in order])
    st.dataframe(cost, use_container_width=True, hide_index=True)

    prop = zd_models.get("TwinShield-Fusion", {})
    base_rates = [zd_models[m]["unknown_detection_rate"] for m in order
                  if m != "TwinShield-Fusion"]
    if prop and base_rates:
        gap = prop["unknown_detection_rate"] - max(base_rates)
        sup = prop["per_class"].get(unknown, {}).get("supervised_alarm_rate", 0.0)
        gt = prop["per_class"].get(unknown, {}).get("gate_alarm_rate", 0.0)
        if gap > 0.05:
            flag(
                f"<b>Measured result.</b> On the quarantined class, the best baseline "
                f"flags <b>{max(base_rates)*100:.1f}%</b> of frames and "
                f"TwinShield-Fusion flags <b>{prop['unknown_detection_rate']*100:.1f}%</b> "
                f"— a gap of {gap*100:.1f} percentage points, at a false-alarm rate of "
                f"{prop['false_positive_rate']*100:.2f}% on normal traffic. "
                f"<br><br>The credit belongs to a specific component and the split is "
                f"printed rather than implied: the supervised head contributes "
                f"<b>{sup*100:.1f}%</b> and the Digital Twin anomaly gate contributes "
                f"<b>{gt*100:.1f}%</b>. The classifier is not what generalises here — "
                f"the residual is.")
        else:
            flag(
                f"<b>Reported as measured, including when it does not favour the "
                f"proposed method.</b> On the quarantined class TwinShield-Fusion "
                f"flags {prop['unknown_detection_rate']*100:.1f}% against the best "
                f"baseline's {max(base_rates)*100:.1f}% — a gap of "
                f"{gap*100:+.1f} percentage points, which is not a meaningful "
                f"advantage. The residual only separates an attack that pushes payload "
                f"bytes outside what the Digital Twin considers normal for that CAN ID; "
                f"where the forged payload sits inside that range there is nothing for "
                f"the gate to fire on. See the per-CAN-ID table below.")

    # ---------------------------------------------------------------- misclassification
    with st.expander(f"▶  What did each model call the {unknown} frames?"):
        rows = []
        for m in order:
            pa = zd_models[m]["per_class"].get(unknown, {}).get("predicted_as", {})
            total = max(sum(pa.values()), 1)
            rows.append({"Model": m,
                         **{k: f"{v:,} ({v/total*100:.0f}%)" for k, v in pa.items()}})
        st.dataframe(pd.DataFrame(rows).fillna("—"), use_container_width=True,
                     hide_index=True)
        st.markdown(f"""
This is the mechanism, not a quirk of training. A supervised classifier assigns
probability mass over the classes in its output layer, and {unknown} is not one of
them. It has no way to say "this is something new" — only to pick the least bad of
the {len(zd['known_classes'])} labels it was given, which for traffic that does not
resemble a flood is "Normal". Adding CNN or LSTM capacity does not change that;
the missing ingredient is a reference for what the bus *should* be doing.
""")

    st.write("")

    # ---------------------------------------------------------------- gate
    section("The anomaly gate, characterised",
            "The channel that does not depend on known attacks: "
            "z = deviation of the Digital Twin residual from its normal behaviour, "
            "standardised per CAN ID and per payload byte.")

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        metric_tile("Threshold", f"z = {gate['threshold']:.2f}",
                    f"set at a {gate['target_fpr']*100:.0f}% target FPR", highlight=True)
    with g2:
        metric_tile("Calibrated on", f"{gate['n_cal_frames']:,}",
                    "held-out NORMAL frames — no attack involved")
    with g3:
        auc_u = gate["auc_vs_normal"].get(unknown)
        metric_tile(f"AUC · {unknown}", f"{auc_u:.4f}" if auc_u else "—",
                    "score alone, normal vs this class",
                    colour=GREEN if (auc_u or 0) > 0.9 else AMBER)
    with g4:
        metric_tile("Fitted on", f"{gate['n_fit_frames']:,}",
                    "normal training frames (mean / σ)")

    stats = gate["score_stats"]
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        classes = [c for c in CLASS_NAMES if c in stats]
        figs = go.Figure()
        for c in classes:
            s = stats[c]
            figs.add_trace(go.Box(
                x=[s["p5"], s["p25"], s["p50"], s["p75"], s["p95"]], y=[c] * 5,
                orientation="h", name=c, marker_color=CLASS_COLORS.get(c, BLUE),
                boxpoints=False, showlegend=False,
                hovertemplate="%{x:.2f}<extra>" + c + "</extra>"))
        figs.add_vline(x=gate["threshold"], line_dash="dash", line_color=VIOLET,
                       annotation_text="gate threshold", annotation_font_color=VIOLET,
                       annotation_font_size=11)
        figs.update_layout(xaxis_title="anomaly score z (log scale)", xaxis_type="log",
                           yaxis_title=None)
        section("Score distribution by true class", small=True)
        st.plotly_chart(style_fig(figs, height=320), use_container_width=True)
        note("Boxes span the 5th–95th percentile of the score on held-out test frames. "
             "A class whose box sits left of the dashed line is one this gate cannot "
             "separate, and that is shown rather than cropped out.")
    with c2:
        sw = gate["sweep"]
        figr = go.Figure()
        for c in attack_classes:
            if c in sw:
                figr.add_trace(go.Scatter(
                    x=sw["fpr"], y=sw[c], mode="lines",
                    name=f"{c} ★" if c == unknown else c,
                    line=dict(color=CLASS_COLORS.get(c, BLUE), width=2.4,
                              dash="solid" if c == unknown else "dot"),
                    hovertemplate="FPR %{x:.2%} → %{y:.1%}<extra>" + c + "</extra>"))
        op = zd_models.get("TwinShield-Fusion", {}).get("per_class", {})
        figr.add_trace(go.Scatter(
            x=[stats["Normal"]["detected"]],
            y=[op.get(unknown, {}).get("gate_alarm_rate", 0.0)],
            mode="markers", name="operating point",
            marker=dict(color=VIOLET, size=12, symbol="diamond"),
            hovertemplate="the threshold actually used<extra></extra>"))
        # Every curve is monotone in the threshold, so clipping the axis to the
        # region anyone would actually operate in hides nothing.
        x_max = max(0.25, min(1.0, stats["Normal"]["detected"] * 4))
        figr.update_layout(xaxis_title="false-positive rate on normal frames",
                           yaxis_title="detection rate", xaxis_tickformat=".0%",
                           yaxis_tickformat=".0%", yaxis_range=[-0.03, 1.05],
                           xaxis_range=[0, x_max],
                           legend=dict(orientation="h", y=1.18, x=0.28))
        section("Detection against false alarms", small=True)
        st.plotly_chart(style_fig(figr, height=320), use_container_width=True)
        note(f"The operating range, so the chosen threshold can be read against the "
             f"alternatives instead of presented as the only possible choice. The axis "
             f"is clipped at {x_max*100:.0f}% false positives — every curve is monotone "
             f"in the threshold and stays flat beyond it.")

    # ---------------------------------------------------------------- per CAN ID
    id_tab = gate.get("residual_by_can_id", [])
    if id_tab:
        section("Why the gate fires — or does not",
                "Mean Digital Twin residual on each attacked CAN ID, normal traffic "
                "against attack traffic. This is the physical quantity behind the score.")
        idt = pd.DataFrame(id_tab)
        figb = go.Figure()
        labels = [f"{r['can_id']}<br>{r['class']}" for _, r in idt.iterrows()]
        figb.add_trace(go.Bar(name="Normal frames on this ID", x=labels,
                              y=idt["mean_residual_normal"], marker_color=GREEN,
                              hovertemplate="%{y:.2f} bytes<extra>Normal</extra>"))
        figb.add_trace(go.Bar(name="Attack frames on this ID", x=labels,
                              y=idt["mean_residual_attack"], marker_color=RED,
                              hovertemplate="%{y:.2f} bytes<extra>Attack</extra>"))
        figb.update_layout(barmode="group", yaxis_title="mean |residual| (bytes)",
                           legend=dict(orientation="h", y=1.16, x=0))
        st.plotly_chart(style_fig(figb, height=290), use_container_width=True)

        idt = idt.assign(ratio=idt["mean_residual_attack"] / idt["mean_residual_normal"])
        weak = idt[idt["ratio"] < 1.5]
        strong = idt[idt["ratio"] >= 1.5]
        msg = ""
        if len(strong):
            msg += ("The twin separates " + ", ".join(
                f"<b>{r['class']}</b> on <code>{r['can_id']}</code> "
                f"({r['mean_residual_normal']:.2f} → {r['mean_residual_attack']:.2f} bytes, "
                f"{r['ratio']:.1f}×)" for _, r in strong.iterrows()) + ". ")
        if len(weak):
            msg += ("<br><br><b>It does not separate</b> " + ", ".join(
                f"<b>{r['class']}</b> on <code>{r['can_id']}</code> "
                f"({r['mean_residual_normal']:.2f} → {r['mean_residual_attack']:.2f} bytes)"
                for _, r in weak.iterrows())
                + ". Those forged payloads sit inside the range the twin already treats "
                "as normal for that ID, so residual magnitude carries no signal and the "
                "gate has nothing to fire on. Detection of that class has to come from "
                "the timing and sequence features the supervised detector also sees. "
                "This is the clearest limit of the current residual representation and "
                "the obvious place for future work.")
        note(msg)

    with st.expander("▶  How the anomaly gate works, precisely"):
        st.markdown(r"""
The gate reads the same Digital Twin residual the detector does, through two views.

**Frame view — how far this one frame departs from its own ECU's normal**, per byte:
""")
        st.latex(r"z_{\text{frame}}(t) \;=\; \max_{b}\; "
                 r"\frac{r_{t,b} - \mu_{i,b}}{\sigma_{i,b}}")
        st.markdown("where $i$ is the frame's arbitration ID and $b$ indexes the 8 "
                    "payload bytes. This is the sharp channel: a forged payload that "
                    "moves one byte far out of range shows up immediately and is not "
                    "averaged away by surrounding traffic.")
        st.markdown("**ECU view — how far a whole ECU stream has drifted** across the "
                    "32-frame context window:")
        st.latex(r"z_{\text{ecu}}(t) \;=\; \max_{s \,\in\, \text{present}}\; "
                 r"\frac{\bar{v}_{s}(t) - \mu_{s}}{\sigma_{s}}")
        st.markdown(f"""
and `score = max(z_frame, z_ecu)`, with an alarm when it exceeds the threshold.

| Quantity | Estimated from |
|---|---|
| μ, σ per CAN ID per byte | {gate['n_fit_frames']:,} **normal training** frames |
| μ, σ per ECU stream | the same frames |
| threshold | {gate['n_cal_frames']:,} **normal validation** frames, at a {gate['target_fpr']*100:.0f}% target FPR |

**No attack frame of any class takes part in fitting or calibration.** That is the
entire basis for calling the gate attack-agnostic: there is no signature in it to
memorise, only a model of normal.

**Why standardise per ID and per byte?** Some ECUs and some bytes — counters, noisy
sensor readings — are intrinsically hard to predict and sit at a high baseline
residual. One global threshold would fire on them constantly. Judging each byte of
each ECU against its own normal behaviour is what makes a single bus-wide threshold
usable at all.
""")

st.write("")
st.markdown("---")

# ===========================================================================
# 2. The Digital Twin
# ===========================================================================
st.markdown("### 🧠 The Digital Twin")
note("A GRU trained on <b>normal CAN traffic only</b>. Given the previous payloads "
     "seen on a CAN ID it predicts the next one; the gap between prediction and "
     "reality is the residual every part of this project is built on. It is never "
     "shown an attack during fitting — which is exactly why an attack it has never "
     "seen still raises its residual.")

use_zd_twin = zeroday_ready()
if use_zd_twin and st.checkbox(
        "Use the fully-trained Digital Twin instead of the zero-day one",
        value=False,
        help="The zero-day twin was refitted without the quarantined attack class. "
             "It is the one the Live Monitor runs, so it is shown by default."):
    use_zd_twin = False

if use_zd_twin:
    residual, expected = get_zd_twin_outputs()
    dt_metrics = get_zero_day()["digital_twin"]
    twin_label = (f"refitted with <b>{get_zero_day()['unknown_class']}</b> held out "
                  "— the twin the Live Monitor runs")
else:
    residual, expected = get_twin_outputs()
    dt_metrics = res["digital_twin"]
    twin_label = "trained on all normal frames in the training split"

df = p.df
y_all = p.windows.y

counts = df.groupby("arbitration_id_hex").size()
normal_counts = df[df["y"] == 0].groupby("arbitration_id_hex").size()
eligible = sorted([i for i in counts.index if normal_counts.get(i, 0) >= 200])
attacked = df[df["y"] != 0].groupby("arbitration_id_hex").size()
default = next((i for i in eligible if attacked.get(i, 0) > 0 and i != "00000000"),
               eligible[0] if eligible else None)

if default is None:
    flag("No CAN ID in this capture carries enough normal traffic to plot.")
else:
    s1, s2, s3 = st.columns([1.1, 1.1, 1.5])
    with s1:
        can_id = st.selectbox("CAN ID", eligible, index=eligible.index(default))

    # Default to whichever byte actually separates attack from normal on this ID --
    # measured per CAN ID rather than a fixed guess that might flatter the result.
    _m = (df["arbitration_id_hex"].to_numpy() == can_id)
    _n, _a = _m & (y_all == 0), _m & (y_all != 0)
    if _a.sum() and _n.sum():
        best_byte = int(np.argmax(residual[_a].mean(axis=0) - residual[_n].mean(axis=0)))
    else:
        best_byte = int(np.argmax(residual[_m].mean(axis=0))) if _m.sum() else 0
    with s2:
        byte_i = st.selectbox("Payload byte", list(range(8)), index=best_byte,
                              format_func=lambda b: f"DATA_{b}"
                              + (" ★ most separating" if b == best_byte else ""))
    with s3:
        max_pts = st.slider("Frames to display", 120, 900, 340, step=20)

    # Frames on this ID from BOTH captures: the tail of the normal recording then
    # the head of the attack one.  A plain first-N slice would return only normal
    # frames and the chart would never show a deviation.
    _all = np.where(_m)[0]
    _src = df["source"].to_numpy()
    _norm_pool, _atk_pool = _all[_src[_all] == "normal"], _all[_src[_all] == "attack"]
    if len(_atk_pool):
        half = min(max_pts // 2, len(_norm_pool))
        sel = np.concatenate([_norm_pool[-half:], _atk_pool[:max_pts - half]])
        boundary = int(half)
    else:
        sel, boundary = _all[:max_pts], None

    actual_b = df[DATA_COLS[byte_i]].to_numpy()[sel].astype(float)
    expect_b = expected[sel, byte_i] * 255.0
    resid_b = np.abs(actual_b - expect_b)
    y_sel = y_all[sel]

    c1, c2 = st.columns(2, gap="large")
    with c1:
        figt = go.Figure()
        figt.add_trace(go.Scatter(y=expect_b, mode="lines", name="Expected",
                                  line=dict(color=CYAN, width=2, dash="dot"),
                                  hovertemplate="expected %{y:.1f}<extra></extra>"))
        figt.add_trace(go.Scatter(y=actual_b, mode="lines", name="Actual",
                                  line=dict(color=TEXT, width=1.6),
                                  hovertemplate="actual %{y:.1f}<extra></extra>"))
        for ci in (1, 2, 3):
            colour = CLASS_COLORS[CLASS_NAMES[ci]]
            pts = np.where(y_sel == ci)[0]
            if len(pts):
                figt.add_trace(go.Scatter(
                    x=pts, y=actual_b[pts], mode="markers", name=CLASS_NAMES[ci],
                    marker=dict(color=colour, size=6, symbol="x"),
                    hovertemplate=f"{CLASS_NAMES[ci]}: %{{y:.1f}}<extra></extra>"))
        if boundary:
            figt.add_vline(x=boundary, line_dash="dash", line_color=MUTED,
                           annotation_text="attack capture begins",
                           annotation_font_color=MUTED, annotation_font_size=10)
        figt.update_layout(yaxis_title=f"DATA_{byte_i} (0–255)", xaxis_title="frame",
                           legend=dict(orientation="h", y=1.16, x=0))
        section("Expected vs Actual",
                f"GRU prediction against the recorded DATA_{byte_i} on {can_id}."
                + (" Frames left of the dashed line come from the normal-driving "
                   "capture." if boundary else ""), small=True)
        st.plotly_chart(style_fig(figt, height=300), use_container_width=True)

    with c2:
        figr2 = go.Figure()
        normal_pts, attack_pts = np.where(y_sel == 0)[0], np.where(y_sel != 0)[0]
        figr2.add_trace(go.Scatter(y=resid_b, mode="lines", name="Residual",
                                   line=dict(color=CYAN, width=1.7),
                                   hovertemplate="%{y:.1f}<extra></extra>"))
        if len(attack_pts):
            figr2.add_trace(go.Scatter(x=attack_pts, y=resid_b[attack_pts],
                                       mode="markers", name="attack frames",
                                       marker=dict(color=RED, size=5),
                                       hovertemplate="%{y:.1f}<extra></extra>"))
        if len(normal_pts):
            figr2.add_hline(y=float(resid_b[normal_pts].mean()), line_dash="dash",
                            line_color=GREEN, annotation_text="mean on normal frames",
                            annotation_font_color=GREEN, annotation_font_size=10)
        if boundary:
            figr2.add_vline(x=boundary, line_dash="dash", line_color=MUTED)
        figr2.update_layout(yaxis_title="bytes", xaxis_title="frame",
                            legend=dict(orientation="h", y=1.16, x=0))
        section("Residual = |Actual − Expected|",
                "The quantity every downstream component reads.", small=True)
        st.plotly_chart(style_fig(figr2, height=300), use_container_width=True)

    if _a.sum() and _n.sum():
        mn_all, ma_all = residual[_n].mean(axis=0) * 255, residual[_a].mean(axis=0) * 255
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Normal frames", x=[f"DATA_{i}" for i in range(8)],
                              y=mn_all, marker_color=GREEN,
                              hovertemplate="%{y:.2f} bytes<extra>Normal</extra>"))
        fig3.add_trace(go.Bar(name="Attack frames", x=[f"DATA_{i}" for i in range(8)],
                              y=ma_all, marker_color=RED,
                              hovertemplate="%{y:.2f} bytes<extra>Attack</extra>"))
        fig3.update_layout(barmode="group", yaxis_title="mean |residual| (bytes)",
                           legend=dict(orientation="h", y=1.2, x=0))
        section("Residual per payload byte",
                f"Every byte on {can_id}, so the byte selector above cannot flatter "
                f"the result.", small=True)
        st.plotly_chart(style_fig(fig3, height=250), use_container_width=True)
        on, oa = float(mn_all.mean()), float(ma_all.mean())
        if oa > on * 1.5:
            note(f"Across all bytes on {can_id} the mean residual rises from "
                 f"<b>{on:.2f}</b> to <b style='color:{GREEN}'>{oa:.2f}</b> bytes under "
                 f"attack — the Digital Twin separates this attack clearly, and the "
                 f"byte selector above cannot flatter that because every byte is shown.")
        else:
            note(f"Across all bytes on {can_id} the mean residual is <b>{on:.2f}</b> on "
                 f"normal frames and <b style='color:{AMBER}'>{oa:.2f}</b> under attack. "
                 f"<b>The residual does not separate this attack on this ID</b>, and that "
                 f"is reported rather than hidden — detection of this class has to come "
                 f"from the timing and sequence features the detector also sees.")

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
note(f"Prediction quality measured on held-out <b>normal</b> frames only — the twin "
     f"models normal behaviour, so that is where its accuracy is meaningful. This twin "
     f"was {twin_label}.")

with st.expander("▶  Architecture, and an honest reading of the MAE"):
    st.markdown(f"""
| Component | Setting |
|---|---|
| Input | last 8 payloads observed on the same CAN ID, scaled to `[0,1]` |
| ID embedding | 16-dim, so one network serves every ECU |
| Recurrent core | GRU, 96 hidden units |
| Head | 64-unit ReLU → 8 outputs → sigmoid |
| Loss | MSE against the true next payload |
| Training data | **Normal frames from the training split only** |

**MAE = {dt_metrics['MAE']:.2f} bytes is not a small number, and it does not need to be.**
CAN payloads carry several independent signals and some bytes — rolling counters,
noisy sensor readings — are inherently hard to predict. What detection needs is not a
residual near zero but a residual that is *reliably different* between normal and
tampered traffic on the same CAN ID. That is what the per-byte chart above lets you
check directly, and it is why the anomaly gate standardises per ID and per byte
rather than thresholding the raw residual.
""")

st.write("")
st.markdown("---")

# ===========================================================================
# 3. Closed-set comparison -- every model on one shared split
# ===========================================================================
st.markdown("### 📈 Closed-set comparison")
note("All attack classes present in training, which is the conventional way this "
     "task is benchmarked. It is reported here as the control for the zero-day "
     "experiment above, not as the headline.")

variants = list(res["results"].keys())
labels = {"with_id": "CAN ID included", "no_id": "CAN ID excluded (controlled)"}
vsel = st.radio("Feature set", variants, horizontal=True,
                format_func=lambda v: labels.get(v, v))
R = res["results"][vsel]
order = [m for m in res["model_order"] if m in R]
support = res["test_class_support"]

note(f"One shared held-out test split ({res['split_sizes']['test']:,} frames). "
     "Class support — " + " · ".join(f"{k}: {v:,}" for k, v in support.items()))

if vsel == "no_id":
    note("In this variant the arbitration ID is removed from the model inputs. The "
         "fusion branch additionally <b>rank-orders</b> its ECU slots by residual "
         "magnitude instead of keeping one fixed slot per CAN ID — otherwise "
         "“slot 1 is active” would be equivalent to “ID 00000000 is present”, "
         "handing the shortcut straight back to the model.")

st.write("")

cards = [m for m in ["CNN", "BiLSTM", "CNN-BiLSTM", "TwinShield-Fusion"] if m in R]
cols = st.columns(len(cards))
best = max(R[m]["metrics"]["Macro F1"] for m in order)
for col, name in zip(cols, cards):
    f1 = R[name]["metrics"]["Macro F1"]
    with col:
        metric_tile(name, f"{f1:.4f}", "Macro F1", highlight=abs(f1 - best) < 1e-9,
                    colour=CYAN if abs(f1 - best) < 1e-9 else None)

st.write("")
section("Model comparison", "Computed from actual predictions on the shared test set.")
table = pd.DataFrame([{
    "Model": m,
    "Accuracy": round(R[m]["metrics"]["Accuracy"], 4),
    "Precision": round(R[m]["metrics"]["Precision (macro)"], 4),
    "Recall": round(R[m]["metrics"]["Recall (macro)"], 4),
    "Macro F1": round(R[m]["metrics"]["Macro F1"], 4),
    "Weighted F1": round(R[m]["metrics"]["Weighted F1"], 4),
} for m in order])


def _shade(col: pd.Series) -> list:
    """Blue gradient over a column without pulling in matplotlib."""
    lo, hi = float(col.min()), float(col.max())
    span = hi - lo
    return [f"background-color: rgba(59,130,246,"
            f"{0.08 + 0.42 * (0.0 if span <= 0 else (float(v) - lo) / span):.3f})"
            for v in col]


st.dataframe(table.style.apply(_shade, subset=["Macro F1"]).format(precision=4),
             use_container_width=True, hide_index=True)

spread = table["Macro F1"].max() - table["Macro F1"].min()
if spread < 0.01:
    note(f"⚠️ <b>All models fall within {spread:.4f} macro-F1 of each other.</b> On this "
         f"feature set the task is effectively <b>saturated</b>, so the differences "
         f"between architectures are not meaningful and should not be read as one "
         f"method beating another."
         + (" This is expected here: arbitration ID alone already separates DoS "
            "perfectly, so most of the label information is available before any "
            "sequence model runs." if vsel == "with_id" else "")
         + " It is also precisely why the zero-day experiment above is the "
           "discriminating one: saturation on a closed set says nothing about what "
           "happens when an attack falls outside it.")

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

figc = go.Figure()
for cls in CLASS_NAMES:
    if cls in pc.columns:
        figc.add_trace(go.Bar(name=cls, x=pc.index, y=pc[cls],
                              marker_color=CLASS_COLORS[cls],
                              hovertemplate="%{y:.4f}<extra>" + cls + "</extra>"))
figc.update_layout(barmode="group", yaxis_title="F1", yaxis_range=[0, 1.05],
                   legend=dict(orientation="h", y=1.15, x=0))
st.plotly_chart(style_fig(figc, height=330), use_container_width=True)

section("Ablation", "Each step adds one component to the same CNN-BiLSTM backbone.")
abl = [m for m in res["ablation_order"] if m in R]
vals = [R[m]["metrics"]["Macro F1"] for m in abl]
acols = st.columns(len(abl) * 2 - 1)
for i, (name, v) in enumerate(zip(abl, vals)):
    with acols[i * 2]:
        metric_tile(name, f"{v:.4f}",
                    "baseline" if i == 0 else f"{v - vals[0]:+.4f} vs baseline",
                    highlight=(i == len(abl) - 1))
    if i < len(abl) - 1:
        with acols[i * 2 + 1]:
            st.markdown(f'<div style="text-align:center;padding-top:2.4rem;'
                        f'color:{MUTED};font-size:1.3rem">→</div>',
                        unsafe_allow_html=True)

figa = go.Figure(go.Scatter(
    x=abl, y=vals, mode="lines+markers+text", text=[f"{v:.4f}" for v in vals],
    textposition="top center", textfont=dict(color=TEXT, size=12),
    line=dict(color=CYAN, width=3), marker=dict(size=11, color=BLUE)))
lo, hi = min(vals), max(vals)
pad = max((hi - lo) * 0.35, 0.004)
figa.update_layout(yaxis_title="Macro F1", yaxis_range=[lo - pad, hi + pad * 1.6])
st.plotly_chart(style_fig(figa, height=290), use_container_width=True)

# ---------------------------------------------------------------- SHAP
st.write("")
section("Why did the model detect this attack?", "SHAP attributions over the "
        "trained detector — computed, not hand-written.")
shap_payload = load_shap()
if shap_payload is None:
    flag("SHAP values not computed yet. Run <code>python compute_shap.py</code> "
         "to generate them.")
else:
    scls = st.selectbox("Explain predictions of class",
                        ["All predictions"] + list(CLASS_NAMES), index=1)
    cidx = None if scls == "All predictions" else CLASS_NAMES.index(scls)
    names, imp = aggregate(shap_payload, cidx, top_k=12)
    if len(names) == 0:
        note(f"The model made no predictions of class “{scls}” in the explained sample.")
    else:
        s1, s2 = st.columns([1.45, 1], gap="large")
        with s1:
            figs2 = go.Figure(go.Bar(
                x=imp[::-1], y=names[::-1], orientation="h",
                marker=dict(color=[CYAN if i == 0 else BLUE
                                   for i in range(len(names))][::-1]),
                hovertemplate="%{x:.5f}<extra></extra>"))
            figs2.update_layout(xaxis_title="mean |SHAP value|", yaxis_title=None)
            st.plotly_chart(style_fig(figs2, height=380), use_container_width=True)
        with s2:
            ecu = shap_payload["ecu_shap"]
            slabels = list(shap_payload["stream_labels"])
            if cidx is not None:
                m_ = shap_payload["pred"] == cidx
                ecu = ecu[m_] if m_.sum() else ecu
            eimp = np.abs(ecu).mean(axis=0)
            eorder = np.argsort(eimp)[::-1][:8]
            fige = go.Figure(go.Bar(
                x=eimp[eorder][::-1], y=[slabels[i] for i in eorder][::-1],
                orientation="h", marker=dict(color=VIOLET),
                hovertemplate="%{x:.5f}<extra></extra>"))
            fige.update_layout(xaxis_title="mean |SHAP| per ECU stream", yaxis_title=None)
            st.plotly_chart(style_fig(fige, height=380,
                                      title="Which ECU stream mattered"),
                            use_container_width=True)
        note("Attributions are summed over the "
             f"{res['config']['window']}-frame context window and averaged over "
             f"{len(shap_payload['pred']):,} explained test frames.")

# ---------------------------------------------------------------- audit
st.write("")
st.markdown("---")
st.markdown("### 🔬 Research validity")
audit = get_audit()
sc_ = audit["can_id_shortcut"]
dup = audit["duplicates"]

a1, a2, a3 = st.columns(3)
with a1:
    metric_tile("CAN ID alone — accuracy", f"{sc_['accuracy']*100:.2f}%",
                "majority-class lookup on ID", colour=RED)
with a2:
    metric_tile("CAN ID alone — macro F1", f"{sc_['macro_f1']:.4f}",
                "same lookup, macro-averaged", colour=AMBER)
with a3:
    metric_tile("Duplicate frames", f"{dup['duplicate_share_pct']:.1f}%",
                f"{dup['test_frames_with_identical_twin_in_train_pct']:.1f}% of test "
                f"frames have a twin in train", colour=AMBER)

per = sc_["per_class_f1"]
flag("<b>Reported openly:</b> a lookup table using nothing but the arbitration ID "
     f"reaches <b>{sc_['accuracy']*100:.2f}% accuracy</b> on this dataset — because every "
     "DoS frame carries ID <code>00000000</code> and no normal frame does "
     f"(DoS F1 = {per['DoS']:.2f}). The same lookup scores "
     f"<b>{per['Brake Tampering']:.2f}</b> and <b>{per['Steering Tampering']:.2f}</b> on "
     "brake and steering tampering, which reuse legitimate IDs. "
     "<br><br>Two consequences, and they shape how this project should be read: "
     "<b>accuracy is not a meaningful headline metric here</b>, and the value of the "
     "Digital Twin residual has to be judged on the tampering classes — and on the "
     "unknown-attack experiment at the top of this page.")

with st.expander("▶  Full dataset audit"):
    st.markdown("**Class balance**")
    st.dataframe(pd.DataFrame(audit["class_balance"]), use_container_width=True,
                 hide_index=True)
    st.markdown("**Duplicate analysis**")
    st.json(dup)
    st.markdown("**CAN ID usage — normal vs attack**")
    st.dataframe(pd.DataFrame(audit["can_id_overlap"]), use_container_width=True,
                 hide_index=True)
    st.markdown(f"""
**Splitting.** Frames are cut into contiguous blocks of 256 and whole blocks are
assigned to train/val/test, stratified by the rarest class each block contains. A
frame's context window never leaves its block, so no frame is shared between splits.
A purely chronological split was rejected because the tampering attacks occupy the
opening ~9% of the attack capture and would leave the test set with none.

**Remaining limitation.** Blocks adjacent in time can still be correlated, and CAN
traffic is highly repetitive ({dup['duplicate_share_pct']:.1f}% of frames are exact
duplicates of another frame). Results are within-capture performance, not evidence of
generalisation to a different vehicle or drive cycle. The zero-day experiment shows
generalisation across *attack classes*, which is a different and weaker claim than
generalisation across vehicles.
""")

# ---------------------------------------------------------------- statement
st.write("")
section("Result statement")
if "CNN-BiLSTM" in R and "TwinShield-Fusion" in R:
    b = R["CNN-BiLSTM"]["metrics"]["Macro F1"]
    t = R["TwinShield-Fusion"]["metrics"]["Macro F1"]
    delta_pp = (t - b) * 100
    if delta_pp > 0.5:
        msg = (f"On the closed set, TwinShield-Fusion achieved a higher macro-F1 than "
               f"the CNN-BiLSTM baseline by <b>{delta_pp:.2f} percentage points</b> "
               f"({b:.4f} → {t:.4f}) on the {labels.get(vsel, vsel)} feature set.")
        tone = GREEN
    elif delta_pp < -0.5:
        msg = (f"On the closed set, TwinShield-Fusion did <b>not</b> improve "
               f"classification performance ({b:.4f} → {t:.4f}, {delta_pp:.2f} pp), "
               f"highlighting the limits of the current residual representation.")
        tone = RED
    else:
        msg = (f"On the closed set, TwinShield-Fusion achieved <b>comparable</b> "
               f"classification performance ({b:.4f} → {t:.4f}, {delta_pp:+.2f} pp) "
               f"while adding Digital Twin residual analysis, adaptive fusion and "
               f"explainability.")
        tone = AMBER

    tamper = [c for c in ("Brake Tampering", "Steering Tampering")
              if c in R["TwinShield-Fusion"]["per_class_f1"]]
    if tamper:
        bt = np.mean([R["CNN-BiLSTM"]["per_class_f1"][c] for c in tamper])
        tt = np.mean([R["TwinShield-Fusion"]["per_class_f1"][c] for c in tamper])
        msg += (f"<br><br>On the two tampering classes — the ones arbitration ID cannot "
                f"separate — mean F1 moved from <b>{bt:.4f}</b> to <b>{tt:.4f}</b> "
                f"({(tt-bt)*100:+.2f} pp).")

    if zeroday_ready():
        zdm = get_zero_day()["models"]
        u = get_zero_day()["unknown_class"]
        if "TwinShield-Fusion" in zdm:
            bl = max(zdm[m]["unknown_detection_rate"] for m in zdm
                     if m != "TwinShield-Fusion")
            pr = zdm["TwinShield-Fusion"]["unknown_detection_rate"]
            msg += (f"<br><br><b>The claim this project actually rests on is the other "
                    f"one.</b> With <b>{u}</b> removed from every training stage, the "
                    f"best baseline flagged <b>{bl*100:.1f}%</b> of its held-out frames "
                    f"and TwinShield-Fusion flagged <b>{pr*100:.1f}%</b>. A closed-set "
                    f"score says how well a model handles attacks someone already "
                    f"labelled; that number says what happens when nobody has.")
            if pr - bl > 0.05:
                tone = GREEN

    st.markdown(f'<div class="ts-card" style="border-color:{tone}55;background:{tone}10">'
                f'<p style="color:{TEXT};font-size:1.0rem;line-height:1.75">{msg}</p>'
                f'</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- technical
st.write("")
with st.expander("▶  Training, saving and loading"):
    import os as _os
    import time as _time
    SAVED_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "saved_models")
    st.markdown("""
Models are trained **offline, before the presentation**, so nothing here is required
during a demo. Training from inside the app is deliberately not offered — it would
risk a multi-minute stall in front of an audience.

```bash
python train.py                        # closed-set experiment, all models
python train_zeroday.py                # unknown-attack experiment (Live Monitor)
python train_zeroday.py --unknown "Steering Tampering"
python compute_shap.py                 # SHAP attributions
python check_demo.py                   # verify every artifact before presenting
```

**Saving** is automatic. **Loading** happens on first page use and is cached for the
session.
""")
    if _os.path.isdir(SAVED_DIR):
        files = sorted(f for f in _os.listdir(SAVED_DIR) if f.endswith((".pt", ".npy", ".npz")))
        st.dataframe(pd.DataFrame([{
            "Artifact": f,
            "Size (KB)": round(_os.path.getsize(_os.path.join(SAVED_DIR, f)) / 1024, 1),
            "Saved": _time.strftime("%Y-%m-%d %H:%M", _time.localtime(
                _os.path.getmtime(_os.path.join(SAVED_DIR, f)))),
        } for f in files]), use_container_width=True, hide_index=True)
    if st.button("↻  Reload models from disk", key="reload_models"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Caches cleared — models and results will be re-read on next render.")

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
