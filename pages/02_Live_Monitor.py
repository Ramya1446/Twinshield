"""
Page 2 -- Live Monitor.  The judge-facing demo, and the whole argument in one screen.

Four models watch the same CAN bus at the same time: the CNN, BiLSTM and
CNN-BiLSTM baselines, and TwinShield-Fusion.  Every one of them was trained by
`train_zeroday.py` with a single attack class quarantined out of every stage --
labels, context windows, the Digital Twin's per-ECU history, and the anomaly
gate's calibration set.

Two replay cases are offered:

  * **Known attack.** An attack class all four models were trained on.  All four
    should catch it, and the page says so if one does not.
  * **Unknown attack.** The quarantined class.  The three baselines are
    supervised classifiers with no output for it, so the best they can do is
    call it Normal.  TwinShield-Fusion carries the Digital Twin residual gate,
    which is fitted on normal traffic only and therefore has an opinion about
    traffic no one labelled.

Nothing on this page is scripted.  On each tick the newly revealed frames go
through four real forward passes and one real gate evaluation, and every badge,
bar, counter and trace shows a number that came back from them.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from preprocessing.loader import CLASS_NAMES
from ui.components import (flag, frame_rows, model_board, note, scoreboard,
                           section)
from ui.data import (data_ready, frame_table, get_demo_cases, get_gate,
                     get_prepared, get_zero_day, id_stream_labels, infer_all,
                     zeroday_notice, zeroday_ready)
from ui.theme import BLUE, CYAN, GREEN, MUTED, RED, VIOLET, style_fig

STEP = 3            # frames revealed per tick
TICK = 0.10         # seconds between ticks
TAIL = 170          # frames kept on the live traces
BASELINES = ("CNN", "BiLSTM", "CNN-BiLSTM")
PROPOSED = "TwinShield-Fusion"
ORDER = (*BASELINES, PROPOSED)

# ---------------------------------------------------------------- header
h1, h2 = st.columns([2.4, 1])
with h1:
    st.markdown("## 🚗 Live Monitor — four models, one bus")
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
if not zeroday_ready():
    zeroday_notice()
    st.stop()

p = get_prepared()
zd = get_zero_day()
gate = get_gate()
known_names = zd["known_classes"]
unknown_name = zd["unknown_class"]
cases = get_demo_cases()
if not cases:
    flag("No replay cases could be built from this dataset.")
    st.stop()

# ---------------------------------------------------------------- case picker
titles = list(cases.keys())
ss = st.session_state
ss.setdefault("case_title", titles[0])

pick_cols = st.columns(len(titles))
for col, title in zip(pick_cols, titles):
    sc = cases[title]
    with col:
        active = ss.case_title == title
        if st.button(title, key=f"case_{sc['key']}", use_container_width=True,
                     type="primary" if active else "secondary"):
            ss.case_title = title
            active = True
        blurb = ("Every model was trained on this attack class."
                 if sc["known"] else
                 f"<b>Quarantined from all four models.</b> No detector on this page "
                 f"has seen a single {unknown_name} frame, in a label, a context "
                 f"window, or the Digital Twin's history.")
        st.markdown(f'<div class="ts-case{" active" if active else ""}">'
                    f'<span class="t">{sc["target"]}</span>{blurb}</div>',
                    unsafe_allow_html=True)

st.write("")
scenario = ss.case_title
sc = cases[scenario]
frames = sc["frames"]
n_frames = len(frames)
lead_n = sc["lead"]
target_name = sc["target"]
y_true = p.windows.y[frames]
is_target = y_true == CLASS_NAMES.index(target_name)

c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    play = st.button("▶  Run comparison", type="primary", use_container_width=True)
with c2:
    pause = st.button("⏸  Pause", use_container_width=True)
with c3:
    reset = st.button("↻  Reset", use_container_width=True)


# ---------------------------------------------------------------- state
def blank() -> None:
    ss.cursor = 0
    ss.playing = False
    ss.hist = {m: {"pred": np.zeros(0, dtype=np.int64),
                   "conf": np.zeros(0, dtype=np.float32),
                   "alarm": np.zeros(0, dtype=bool),
                   "gate_alarm": np.zeros(0, dtype=bool)} for m in ORDER}
    ss.gscore = np.zeros(0, dtype=np.float32)
    ss.gchannel = np.zeros(0, dtype="<U5")
    ss.gbyte = np.zeros(0, dtype=np.int64)
    ss.gstream = np.zeros(0, dtype=np.int64)
    ss.alpha = np.zeros((0, len(id_stream_labels())), dtype=np.float32)


if ss.get("sc_key") != scenario:
    ss.sc_key = scenario
    blank()
if play:
    blank()
    ss.playing = True
if pause:
    ss.playing = False
if reset:
    blank()

# ---------------------------------------------------------------- inference
# Only the frames revealed on this tick are pushed through the four models.
run_lo, run_hi = None, None
if ss.playing and ss.cursor < n_frames:
    run_lo, run_hi = ss.cursor, min(ss.cursor + STEP, n_frames)
elif ss.cursor == 0 and not ss.playing:
    run_lo, run_hi = 0, min(12, n_frames)      # idle: never show a blank console

if run_lo is not None:
    out = infer_all(frames[run_lo:run_hi])
    g = out["gate"]
    ss.gscore = np.concatenate([ss.gscore, g["score"]])
    ss.gchannel = np.concatenate([ss.gchannel, g["channel"].astype("<U5")])
    ss.gbyte = np.concatenate([ss.gbyte, g["top_byte"]])
    ss.gstream = np.concatenate([ss.gstream, g["top_stream"]])
    a = out[PROPOSED]["attention"]
    ss.alpha = np.concatenate([ss.alpha, a]) if len(ss.alpha) else a
    for m in ORDER:
        r = out[m]
        h = ss.hist[m]
        for k, v in (("pred", r["pred"]), ("conf", r["conf"]),
                     ("alarm", r["alarm"]), ("gate_alarm", r["gate_alarm"])):
            h[k] = np.concatenate([h[k], v])
    ss.cursor = run_hi

shown = len(ss.gscore)
idx = frames[:shown]
in_attack = shown > lead_n
target_seen = int(is_target[:shown].sum())
target_total = int(is_target.sum())
is_normal = y_true == 0

# ---------------------------------------------------------------- board
cards = []
for m in ORDER:
    h = ss.hist[m]
    live = bool(h["alarm"][-1]) if shown else False
    pred_name = known_names[int(h["pred"][-1])] if shown else "Normal"
    conf = float(h["conf"][-1]) if shown else 0.0
    gate_live = bool(h["gate_alarm"][-1]) if shown else False

    # Caught = raised an alarm on a frame whose true class is this case's attack.
    caught = int((h["alarm"] & is_target[:shown]).sum()) if shown else 0
    mark = ("hit" if caught else "missed") if target_seen else ""

    # The headline is the standing verdict, not just the current frame.  Attack
    # frames are sparse, so a purely instantaneous readout would show every model
    # as quiet a second after the one frame that mattered went past.
    if live and gate_live and pred_name == "Normal":
        verdict, meta = "🔵 UNKNOWN ANOMALY", (
            "No class fits — flagged by deviation from learnt-normal behaviour.")
    elif live:
        verdict, meta = "🔴 ATTACK DETECTED", (
            f"Classified as <b>{pred_name}</b> · {conf*100:.0f}% confident")
    elif caught:
        verdict, meta = "✅ ATTACK FLAGGED", (
            f"Alerted on <b>{caught}</b> of the {target_seen} {target_name} frames "
            f"so far · bus quiet on the current frame")
    elif target_seen:
        verdict, meta = "❌ MISSED THE ATTACK", (
            f"Called all {target_seen} {target_name} frames <b>{pred_name}</b> — "
            f"{conf*100:.0f}% confident on the current one")
    else:
        verdict, meta = "🟢 NO ALERT", (
            f"Says <b>{pred_name}</b> · {conf*100:.0f}% confident")

    card = {"name": m, "tag": "PROPOSED" if m == PROPOSED else "BASELINE",
            "proposed": m == PROPOSED, "alarming": live, "verdict": verdict,
            "meta": meta, "conf": conf, "mark": mark,
            "caught": caught, "target": target_seen, "target_name": target_name,
            # A detector that alarms on everything would catch every attack, so
            # the false-alarm count is shown right beside the catch count.
            "false": int((h["alarm"] & is_normal[:shown]).sum()) if shown else 0,
            "normal": int(is_normal[:shown].sum())}
    if m == PROPOSED and shown:
        z = float(ss.gscore[-1])
        card["gate"] = (f'Digital Twin gate&nbsp; z = <b>{z:.1f}</b> '
                        f'/ threshold {gate.threshold:.1f}')
        card["gate_fired"] = gate_live
    cards.append(card)

model_board(cards)
note(f"Replaying <b>{shown:,}</b> of {n_frames:,} frames · "
     + ("lead-in from the normal-driving capture" if not in_attack
        else f"attack capture · {target_seen} of {target_total} {target_name} "
             f"frames revealed so far")
     + f" · {sc['test_share']}% of this replay is held-out test data.")

st.write("")

# ---------------------------------------------------------------- verdict strip
if target_seen:
    caught = {m: int((ss.hist[m]["alarm"] & is_target[:shown]).sum()) for m in ORDER}
    base_hits = [m for m in BASELINES if caught[m]]
    if sc["known"]:
        if len(base_hits) == len(BASELINES) and caught[PROPOSED]:
            scoreboard(
                f"All four models detected {target_name}.",
                f"This attack class was in every model's training set, so this is the "
                f"expected result and it is the control for the other case — it shows "
                f"the baselines are trained properly and are not failing for some "
                f"unrelated reason. Frames of {target_name} caught: "
                + " · ".join(f"<b>{m}</b> {caught[m]}/{target_seen}" for m in ORDER)
                + ".", tone="win")
        else:
            missed = [m for m in ORDER if not caught[m]]
            scoreboard(
                f"{len(ORDER) - len(missed)} of {len(ORDER)} models detected {target_name}.",
                "Reported as measured: " + ", ".join(f"<b>{m}</b> raised no alarm on "
                f"any of the {target_seen} frames revealed" for m in missed) + ".",
                tone="flat")
    else:
        if caught[PROPOSED] and not base_hits:
            scoreboard(
                f"The three baselines missed {target_name}. TwinShield-Fusion caught it.",
                f"None of the four models was trained on a single {target_name} frame. "
                f"The baselines are supervised classifiers over raw CAN traffic — with "
                f"no output class for this attack, the closest thing they can say is "
                f"“Normal”, and that is what all three said on every one of the "
                f"{target_seen} frames revealed so far. TwinShield-Fusion raised "
                f"<b>{caught[PROPOSED]}</b> of {target_seen} because the Digital Twin "
                f"residual measures <i>deviation from normal</i>, which is defined "
                f"whether or not anybody has ever labelled this attack.", tone="win")
        elif caught[PROPOSED]:
            scoreboard(
                f"TwinShield-Fusion caught {caught[PROPOSED]}/{target_seen} "
                f"{target_name} frames; some baselines also alarmed.",
                "Shown as measured: "
                + " · ".join(f"<b>{m}</b> {caught[m]}/{target_seen}" for m in ORDER)
                + ". A baseline alarm here is a misclassification into one of the "
                "classes it <i>does</i> know rather than genuine recognition, and the "
                "frame table below shows what each one actually called the traffic.",
                tone="flat")
        else:
            scoreboard(
                f"No model detected {target_name} on the frames revealed so far.",
                "This is the measured outcome and it is not dressed up. The Digital "
                "Twin residual only separates an attack that pushes payload bytes "
                "outside what the twin considers normal for that CAN ID; where the "
                "forged payload sits inside that range, the gate has nothing to fire "
                "on. The Results page quantifies exactly that, per attack class.",
                tone="flat")
    st.write("")

# ---------------------------------------------------------------- live traces
left, right = st.columns([1.3, 1], gap="large")

with left:
    section("CAN Traffic Stream", "As classified by TwinShield-Fusion.", small=True)
    if shown:
        h = ss.hist[PROPOSED]
        table = frame_table(idx, h["pred"], h["conf"], names=known_names,
                            flagged=h["gate_alarm"])
        frame_rows(table, limit=13)
    else:
        st.caption("Press ▶ Run comparison to begin.")

with right:
    section("Digital Twin anomaly score", small=True)
    if shown:
        tail = slice(max(0, shown - TAIL), shown)
        z = ss.gscore[tail]
        tgt = np.where(is_target[:shown][tail])[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=z, mode="lines", line=dict(color=CYAN, width=2), name="anomaly score z",
            hovertemplate="z = %{y:.2f}<extra></extra>"))
        if len(tgt):
            fig.add_trace(go.Scatter(
                x=tgt, y=z[tgt], mode="markers", marker=dict(color=RED, size=6),
                name=f"true {target_name} frames",
                hovertemplate="z = %{y:.2f}<extra></extra>"))
        # A short annotation: at this height a long one lies across the trace,
        # and the note under the chart carries the full explanation anyway.
        fig.add_hline(y=gate.threshold, line_dash="dash", line_color=VIOLET,
                      annotation_text="gate threshold",
                      annotation_position="top left",
                      annotation_font_color=VIOLET, annotation_font_size=10)
        fig.update_layout(yaxis_title="z (deviation from normal)", xaxis_title="frame",
                          legend=dict(orientation="h", y=1.18, x=0))
        st.plotly_chart(style_fig(fig, height=236), use_container_width=True,
                        key=f"gate_{shown}")
        note(f"The dashed line is the alarm threshold, set at a "
             f"{gate.target_fpr*100:.0f}% false-positive rate on "
             f"{gate.n_cal_frames:,} held-out <b>normal</b> frames before any attack "
             f"was involved — nothing in it was tuned to the attack you are watching.")

    section("Where the deviation is", small=True)
    if shown:
        labels = id_stream_labels()
        ch = str(ss.gchannel[-1])
        if ch == "frame":
            where = (f"payload byte <b>DATA_{int(ss.gbyte[-1])}</b> on CAN ID "
                     f"<b>{p.df['arbitration_id_hex'].to_numpy()[idx[-1]]}</b>")
        else:
            si = int(ss.gstream[-1])
            where = (f"ECU stream <b>{labels[si] if si < len(labels) else si}</b> "
                     f"across the 32-frame window")
        st.markdown(f'<div class="ts-note">Current frame scores <b>z = '
                    f'{float(ss.gscore[-1]):.2f}</b>, driven by {where}.</div>',
                    unsafe_allow_html=True)

        a = ss.alpha[-1]
        order = np.argsort(a)[::-1][:6]
        fig2 = go.Figure(go.Bar(
            x=a[order][::-1], y=[labels[i] for i in order][::-1], orientation="h",
            marker=dict(color=[BLUE if i != order[0] else CYAN for i in order][::-1]),
            hovertemplate="%{x:.3f}<extra></extra>"))
        fig2.update_layout(xaxis_title="adaptive fusion attention weight", yaxis_title=None)
        st.plotly_chart(style_fig(fig2, height=206), use_container_width=True,
                        key=f"att_{shown}")

# ---------------------------------------------------------------- what is replayed
with st.expander("▶  What exactly is being replayed, and what was held out?"):
    comp = ", ".join(f"{k}: {v:,}" for k, v in sc["composition"].items())
    lr, ar = sc["lead_range"], sc["attack_range"]
    ss_ = zd["split_sizes"]
    st.markdown(f"""
**Two real recordings, played back to back — not one continuous drive.**

| Segment | Frames | Source |
|---|---|---|
| Lead-in (first {lead_n}) | {lr[0]:,}–{lr[1]-1:,} | normal-driving capture |
| Attack (next {n_frames - lead_n}) | {ar[0]:,}–{ar[1]:,} | attack capture |

This join is necessary and is stated plainly rather than disguised: in GEM-CAN the
attack capture is **already under attack from its first frame** — brake tampering
starts at t = 8.2 s and repeats every ~11 s — so no stretch of that recording runs
clean and then turns malicious. The lead-in supplies genuine pre-attack traffic.

Every frame is real and unmodified, and each is classified from **its own** true
preceding context, so no prediction depends on the join.

- **Composition:** {comp}
- **Held-out share:** {sc['test_share']}% of these frames are in the test split.
- **Per tick:** {STEP} frames are revealed, then four real forward passes and one
  real gate evaluation run on them.

**How “{unknown_name}” was quarantined.** `train_zeroday.py` removed it from every
training stage before any model was fitted:

| Stage | What was removed |
|---|---|
| Detector labels | the class has no index in the {len(known_names)}-way output space at all |
| Detector inputs | {ss_['dropped_contaminated_train']:,} train and {ss_['dropped_contaminated_val']:,} val frames were dropped because their 32-frame context window contained one |
| Digital Twin | fitted only on normal frames whose per-ECU history contained none |
| Anomaly gate | mean/σ from normal training frames; threshold from normal validation frames |

The test split was left **untouched** — it still holds
{zd['test_class_support'].get(unknown_name, 0):,} {unknown_name} frames, which is
what makes the comparison meaningful.

**What each model sees.** All three baselines read the raw CAN window
(payload, DLC, timing, arbitration ID). TwinShield-Fusion reads the same window
plus the 8-dimensional Digital Twin residual, and additionally runs the anomaly
gate. That gate is the only component on this page whose decision does not depend
on having seen the attack before.
""")

# ---------------------------------------------------------------- drive the loop
if ss.playing and ss.cursor < n_frames:
    time.sleep(TICK)
    st.rerun()
elif ss.playing:
    ss.playing = False
    st.rerun()
