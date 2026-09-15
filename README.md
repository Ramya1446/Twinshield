# 🛡️ TwinShield-Fusion

**A Digital Twin-Assisted Adaptive Multi-ECU Residual Fusion Framework for Explainable Intrusion Detection in CAN Networks**

Final-year CSE research project — CAN-bus intrusion detection on **real** electric-vehicle traffic.

```
                                           ┌→ CNN-BiLSTM ───→ names a known attack ─┐
Real CAN Traffic → Digital Twin → Residual ┤                                        ├→ Detection → SHAP
                                           └→ Anomaly Gate ─→ flags an unknown one ─┘
```

The second branch is the point. A supervised classifier can only emit a label it was
trained on; the Digital Twin is fitted on **normal traffic only**, so its residual measures
deviation from normal and is defined for an attack nobody has ever labelled.

Every number the application displays — accuracy, F1, confidence, residuals, attention weights,
SHAP values, and the CAN frames themselves — is computed from the real GEM-CAN dataset or from a
trained model checkpoint. Nothing is simulated, hard-coded, or fabricated.

---

## 1. Quick start

```bash
pip install -r requirements.txt
```

```bash
python train.py            # closed-set experiment: every model, every attack class
python train_zeroday.py    # unknown-attack experiment: one class quarantined
python compute_shap.py     # SHAP attributions
python check_demo.py       # verify every artifact before presenting
streamlit run app.py
```

Train once before the presentation. The app then loads saved models and starts instantly —
judges never wait for training. `check_demo.py` verifies every artifact and runs one real
inference through all four models, so a missing file surfaces before you are standing in
front of the panel.

**Three pages:** *Overview* → *Live Monitor* → *Results*.

---

## 2. Dataset setup

The project uses **GEM-CAN**, real CAN traffic captured from a GEM e6 autonomous electric
vehicle on a closed track (CAN @ 250 kbps).

| | |
|---|---|
| Source | <https://zenodo.org/records/17834776> |
| File | `GEM_CAN_Dataset.zip` (1.0 MB, MD5 `02cd463677e37cf9becdd4374e9db26c`) |
| Licence | CC-BY-NC-3.0 (non-commercial academic use) |

Download and extract so the CSVs land here:

```
data/GEM_CAN_Dataset/
├── GEM_Normal_Driving_Raw.csv
├── GEM_Attack_Scenario_Raw.csv
├── GEM_Attack_Scenario_Raw_Labeled.csv
└── data_preprocessing.py
```

Or point elsewhere with `set GEMCAN_DIR=D:\path\to\GEM_CAN_Dataset`.

### What is actually in it

| Class | Frames | CAN IDs used |
|---|---:|---|
| Normal | 101,039 | 13 distinct IDs |
| DoS | 41,080 | `00000000` only |
| Brake Tampering | 663 | `18FF00F9` (also used by normal traffic) |
| Steering Tampering | 662 | `18FF70F9` (also used by normal traffic) |
| **Total** | **143,444** | |

The loader adapts to the files as published, including the corrupted `tiTestaTp` header in the
two attack CSVs and the trailing blank line in the normal capture. Only the four classes above
are ever claimed, because they are the only ones present.

---

## 3. Training

```bash
python train.py                       # full run: both feature sets, ~45–60 min on CPU
python train.py --epochs 3            # fast run for a rehearsal
python train.py --variants no_id      # only the ID-free controlled experiment
```

Trains, on one shared leakage-controlled split:

| Model | Input |
|---|---|
| CNN | raw CAN window |
| BiLSTM | raw CAN window |
| CNN-BiLSTM | raw CAN window |
| + Digital Twin | raw + scalar residual magnitude |
| GRU Digital Twin + CNN-BiLSTM (`+ Residual`) | raw + 8-byte residual vector |
| **TwinShield-Fusion** | raw + residual + adaptive multi-ECU attention |

Everything runs **twice** — once with the arbitration ID available and once with it removed
(see §7). Weights → `saved_models/`, metrics → `results/results.json`, console log →
`results/train_log.txt`.

### The unknown-attack (zero-day) run

```bash
python train_zeroday.py                              # Brake Tampering held out
python train_zeroday.py --unknown "Steering Tampering"
python train_zeroday.py --epochs 3                   # quick rehearsal run
python train_zeroday.py --target-fpr 0.005           # tighter alarm threshold
```

This is the experiment the Live Monitor demonstrates. One attack class is designated
*unknown* and removed from **every** fitting stage — not just from the labels:

| Stage | What is removed |
|---|---|
| Detector label space | the class has no output index at all; the heads are 3-way, not 4-way |
| Detector inputs | every train/val frame whose 32-frame context window contains one |
| GRU Digital Twin | refitted from scratch on normal frames whose per-ECU history contains none |
| Anomaly gate | μ/σ from normal **training** frames; threshold from normal **validation** frames |

The test split is left untouched and still contains the class. All four models — CNN,
BiLSTM, CNN-BiLSTM and TwinShield-Fusion — are then asked the only question comparable
across different label spaces: **did you raise an alarm?**

Artifacts land under `saved_models/zeroday__*` and `results/zero_day.json`. Nothing here
overwrites the closed-set run, so the two experiments coexist.

### Saving and loading

Saving is automatic. The app loads through `ui/data.py`:

```python
from ui.data import get_detector, get_digital_twin, infer
model = get_detector("with_id")            # closed-set TwinShield-Fusion
twin  = get_digital_twin()                 # GRU Digital Twin
proba, attention = infer(frame_indices)    # real forward pass

from ui.data import get_gate, get_zd_models, infer_all
out = infer_all(frame_indices)             # all four zero-day models + the gate
out["TwinShield-Fusion"]["alarm"]          # supervised head OR anomaly gate
out["gate"]["score"], get_gate().threshold # the unknown-attack channel
```

Closed-set checkpoints are named `{variant}__{arch}__{feature_mode}.pt`, e.g.
`with_id__twinshield__fusion.pt`. Zero-day artifacts are prefixed `zeroday__` —
`zeroday__digital_twin.pt`, `zeroday__cnn.pt`, `zeroday__gate.npz` and so on. Delete
`saved_models/prepared_dataset.pkl` to force the dataset cache to rebuild.

---

## 4. Demo instructions

**Route:** Overview → *START LIVE DEMO* → Live Monitor (both test cases) → Results.
About 3 minutes.

1. **Overview** — the pipeline, the dataset card, and the headline result: how many of the
   held-out unknown-attack frames each of the four models flagged.
2. **Live Monitor** — four models watching the same bus, side by side, with two replay cases.
   - **Test Case 1 · Known Attack (DoS).** Every model was trained on it. All four should
     light up. This is the control: it shows the baselines work.
   - **Test Case 2 · Unknown Attack (Brake Tampering).** Quarantined from all four. The three
     baselines end the replay reading **MISSED THE ATTACK**; TwinShield-Fusion reads
     **ATTACK FLAGGED**, and its card shows the gate's live `z` against the threshold.

   Press **▶ Run comparison**, let it play, **⏸ Pause** to talk over a frozen screen. Each card
   carries a catch count *and* a false-alarm count, so the comparison cannot be read as a
   detector that simply alarms on everything.
3. **Results** — the whole unknown-attack experiment (alarm rate per class, the cost table, the
   gate's score distribution and ROC, per-CAN-ID residuals), then the Digital Twin itself, then
   the closed-set comparison, ablation, SHAP and the dataset audit.

**Before you present:** run `python train.py`, `python train_zeroday.py` and
`python compute_shap.py`, then `python check_demo.py`, then open the app once so the caches
are warm. Have `results/zero_day.json` handy as a backup.

---

## 5. What to say on each page

**Overview (~25 s)**
> "This detects cyberattacks on the CAN bus of an electric vehicle, using 143,000 real frames
> from a GEM e6. Most published CAN intrusion detectors are trained on a fixed catalogue of
> attacks — and a car on the road meets attacks that were not in anybody's catalogue. So we
> train a Digital Twin on normal traffic only. It learns what the car *should* be saying, and
> anything that deviates raises a residual — whether or not that attack was ever labelled."

**Live Monitor — Test Case 1, known attack (~40 s)**
> "Four models are watching the same bus: three standard baselines and ours. This is a DoS
> flood, and all four were trained on it. Every one of them catches it — that is the control,
> it shows the baselines are properly trained and not failing for some unrelated reason."

**Live Monitor — Test Case 2, unknown attack (~70 s)**
> "Now the same four models, and the same bus, but this attack class was removed from every
> training stage — from the labels, from the context windows, from the twin's history, from the
> gate's calibration. None of these models has ever seen a brake-tampering frame.
>
> The three baselines are supervised classifiers. They have no output neuron for this attack, so
> the closest thing they can say is 'Normal' — and that is what all three say, confidently, on
> every frame of it. Ours flags it, and the card tells you which component did: not the
> classifier, the Digital Twin residual. Deviation from normal is defined whether or not anybody
> ever labelled this attack. That is the contribution."

**Results (~60 s)**
> "Here is that experiment measured rather than demonstrated: alarm rate per class for all four
> models, and the cost — our false-alarm rate on normal traffic, right next to it, because a
> detector that alarms on everything would score 100%. This is the gate's whole operating range,
> so you can see the threshold we picked against the alternatives.
>
> And here is what we are upfront about: the residual separates brake tampering strongly, and
> steering tampering not at all — the forged steering payloads sit inside the range the twin
> already considers normal. Below that is the conventional closed-set benchmark, where CAN ID
> alone already gets 98.9% accuracy because every DoS frame uses ID zero. That is exactly why we
> report macro-F1, re-run everything without the ID, and why the unknown-attack experiment is
> the one that actually discriminates between methods."

---

## 6. Measured results

### 6a. Unknown-attack experiment — the claim this project rests on

Produced by `python train_zeroday.py`; the app reads it from `results/zero_day.json` and
every figure below is printed live from that file. **The table here is the shape of the
result, not a transcription** — fill it from your own run rather than quoting this README,
because the numbers depend on the epoch count and seed you train with:

| Model | Detects the quarantined class | False alarms on normal | Known-class macro-F1 |
|---|---:|---:|---:|
| CNN | — | — | — |
| BiLSTM | — | — | — |
| CNN-BiLSTM | — | — | — |
| **TwinShield-Fusion** | — | — | — |

What the experiment is designed to show, and what it is not:

- **The three baselines are expected to score ~0%**, and that is not a failure of training —
  it is the definition of a closed-set classifier. Check the "known-class macro-F1" column:
  they handle the classes they *were* trained on perfectly well. The `check_demo.py` output
  prints both numbers side by side for exactly this reason.
- **The detection has to be paid for.** TwinShield-Fusion's false-alarm rate on normal
  traffic will be higher than the baselines', because the anomaly gate is an extra alarm
  source. The app prints both, and the ROC sweep on the Results page shows the whole
  trade-off rather than one flattering point.
- **The credit belongs to the residual, not the classifier.** The Results page splits
  TwinShield-Fusion's detection rate into "from supervised head" and "from anomaly gate". On
  a properly quarantined class the supervised contribution should be ~0 — it has no label to
  use. Claiming the *architecture* generalises would be wrong; the *Digital Twin residual*
  does.
- **It does not work on every class.** Brake tampering is separated strongly by the residual
  (§7d); steering tampering is not separated at all. Run
  `python train_zeroday.py --unknown "Steering Tampering"` and the app will report a
  near-zero result for the proposed model too, without dressing it up. That negative case is
  worth showing: it is the honest boundary of the method.

### 6b. Closed-set comparison — the control

All models, one shared held-out test split (21,588 frames; Normal 15,063 · DoS 6,288 ·
Brake Tampering 119 · Steering Tampering 118). Macro-F1:

| Model | CAN ID included | CAN ID excluded |
|---|---:|---:|
| CNN | 0.9968 | 0.9914 |
| BiLSTM | 0.9948 | 0.9895 |
| CNN-BiLSTM | 0.9979 | 0.9914 |
| GRU Digital Twin + CNN-BiLSTM | 0.9989 | 0.9979 |
| **TwinShield-Fusion** | 0.9979 | **0.9979** |

Ablation (same CNN-BiLSTM backbone, one component added per step):

| Stage | CAN ID included | CAN ID excluded |
|---|---:|---:|
| CNN-BiLSTM | 0.9979 | 0.9914 |
| + Digital Twin | 0.9958 | 0.9965 |
| + Residual | 0.9989 | 0.9979 |
| + Adaptive Fusion | 0.9979 | 0.9979 |

**How to read this, stated plainly:**

- **With CAN ID, the task is saturated.** Every model sits within 0.004 macro-F1 of every
  other, so no architecture can be said to beat another. TwinShield-Fusion is *comparable*
  to the baseline, not better, and the app reports it that way.
- **Without CAN ID, the Digital Twin residual does real work**: CNN-BiLSTM 0.9914 → 0.9979,
  a gain of 0.65 percentage points, and the ablation rises monotonically.
- **Adaptive fusion adds nothing on top of the residual** (0.9979 → 0.9979). The gain in this
  experiment comes from the residual, not from the attention layer. That is the measured
  result and it is not dressed up as anything else.
- The ID-free fusion model is trained with **rank-ordered** ECU slots. With fixed per-ID slots
  it scored 0.9989, but that number was inflated: slot position leaked the arbitration ID.
  0.9979 is the leak-free figure and the one reported.
- **Saturation is the reason §6a exists.** When every architecture lands within 0.004 macro-F1
  of every other, a closed-set benchmark has stopped discriminating between methods — it is
  measuring how easy the dataset is, not how good the model is. The unknown-attack experiment
  asks a question the closed set cannot: what happens when the attack falls outside the label
  set entirely. That is where the four models separate, and it is not a small gap.

## 7. Research validity — reported, not hidden

The audit (`evaluation/audit.py`, shown in-app under *Research validity*) found three things that
materially affect how results should be read:

**a) The CAN ID shortcut.** A lookup table using only the arbitration ID scores **98.9% accuracy**
but **0.498 macro-F1** on the test split — F1 = 1.00 on DoS, **0.00** on both tampering classes.
Every DoS frame uses ID `00000000` and no normal frame does, so DoS is trivially separable without
any learning. Consequences:

- **Accuracy is not a meaningful headline metric on this dataset.** Macro-F1 is.
- The Digital Twin residual must be judged on **brake and steering tampering**, which reuse
  legitimate IDs and cannot be separated by ID at all.
- Every model is additionally trained with arbitration ID **excluded** (`no_id`), selectable in
  the app, so the shortcut can be removed entirely.

**b) Duplicate frames.** 87.4% of frames are byte-identical to another frame, and 80.6% of test
frames have an exact twin in train. This is intrinsic to CAN — ECUs repeat fixed status payloads —
but it inflates any frame-level metric, and is another reason not to lead with accuracy.

**c) Splitting.** Frames are cut into contiguous 256-frame blocks; whole blocks go to
train/val/test, stratified by the rarest class each contains. A frame's context window never
leaves its block, so no frame is shared across splits.

A purely chronological split was **rejected**: the tampering attacks occupy the opening ~9% of the
attack capture, so a chronological cut leaves the test set with zero tampering frames and the
rarest classes become unmeasurable.

**d) The residual works on brake tampering, not on steering tampering.** Measured on the trained
twin, per CAN ID:

| CAN ID | Class | Mean \|residual\| | Verdict |
|---|---|---:|---|
| `18FF00F9` | Normal | 2.58 bytes | |
| `18FF00F9` | Brake Tampering | **23.02 bytes** | 8.9× rise — clearly separated |
| `18FF70F9` | Normal | 6.39 bytes | |
| `18FF70F9` | Steering Tampering | **5.28 bytes** | *lower* than normal — not separated |

For brake tampering the twin localises the attack precisely: bytes `DATA_4` and `DATA_5` move from
~0.5 to 58.8 and 77.5. For steering tampering the forged payloads sit inside the range the twin
already considers normal for that ID, so residual magnitude carries no signal and detection of that
class must come from the timing and sequence features the detector also receives.

This is stated plainly on the Digital Twin page, which recomputes the comparison live for whichever
CAN ID is selected and prints the negative result when it occurs. It is the clearest honest limit
of the current residual representation and the obvious place for future work.

**e) The zero-day protocol, and what it does and does not prove.** Quarantining an attack class
by deleting its label is not enough — a Normal frame whose 32-frame context window contains
injected brake frames has still shown the detector what the attack looks like, and a Normal frame
whose per-ECU history contains them has still shown the Digital Twin. `train_zeroday.py`
therefore excludes both, refits the twin from scratch, and prints how many frames that cost
(the app shows the counts too). The test split is never touched.

What this establishes is **generalisation across attack classes within one capture**. It is not
generalisation across vehicles, drive cycles or attack families — those would need a second
dataset. The claim made in the app is the narrow one.

A second honest caveat: with only one capture, the quarantined class is "unknown" to the model
but its *CAN ID* is not. `18FF00F9` carries legitimate traffic too, so the twin has a normal
profile for it, which is precisely what the residual is measured against. An attack arriving on
an ID the twin has never seen at all would fall back on the unknown-ID embedding and is a
weaker case — not tested here, and not claimed.

**Remaining limitation.** Blocks adjacent in time remain correlated, and the capture is a single
vehicle on a single closed track. Results are within-capture performance and are not evidence of
generalisation to another vehicle or drive cycle.

---

## 8. Likely judge questions

**Q: Is this real data or simulated?**
Real. GEM-CAN from Zenodo, 143,444 frames off a GEM e6. The replay streams recorded frames in
recorded order; we generate no CAN traffic.

**Q: 99% accuracy — isn't that too good?**
Yes, and we say so on the Results page. A lookup on CAN ID alone gets 98.9% because DoS uses a
unique ID. That's why we report macro-F1 and per-class F1, and why we re-run everything without
the ID.

**Q: Your models all score ~0.99 — doesn't that mean TwinShield-Fusion adds nothing?**
On the *CAN ID included* feature set, yes, and the app says so rather than spinning it. Every
model lands within ~0.004 macro-F1 of every other, which means the task is **saturated**, not that
one architecture beat another. That is the direct consequence of finding (a) in §7: arbitration ID alone
already carries almost all the label information, so there is nothing left for a sequence model to
contribute. The Results page prints a saturation warning whenever the spread is under 0.01 and the
closing statement reports "comparable" instead of claiming a win. The *CAN ID excluded* variant is
the experiment that can actually discriminate between methods.

**Q: What does the Digital Twin actually add?**
It is the only component that can catch tampering on a *legitimate* CAN ID. DoS is separable by ID;
brake and steering tampering are not. Compare the tampering columns in per-class F1, and the
ablation ladder.

**Q: Why a GRU for the twin and a CNN-BiLSTM for the detector?**
The twin does short-horizon regression per ECU — a GRU is cheap and sufficient. The detector needs
local burst patterns (convolutions) and longer-range order (BiLSTM). Both baselines are reported
separately so the choice is evidenced, not asserted.

**Q: How do you know there's no data leakage?**
Block-wise splitting with in-block context windows, so no frame appears in two splits. We also
report the duplicate rate (87.4%) rather than hiding it — see §7.

**Q: Is the attention real or decoration?**
Real. It is the softmax over learnt additive-attention scores across per-ECU residual streams,
returned by `forward_with_attention` and plotted directly.

**Q: Are the SHAP values real?**
Yes — `shap.GradientExplainer` against the trained PyTorch model, on frames sampled from the test
split, stratified by class. Computed by `compute_shap.py`, cached to `results/shap.npz`.

**Q: Does the Digital Twin catch every attack?**
No, and we show where it doesn't. It separates brake tampering strongly (residual 2.58 → 23.02
bytes) but **not** steering tampering (6.39 → 5.28, i.e. lower than normal). The forged steering
payloads fall inside the range the twin already treats as normal for that ID. That class is caught
using the timing and sequence features instead. Improving the residual representation for it is the
clearest next step — see §7(d).

**Q: Why frame-level rather than window-level classification?**
Brake and steering tampering are injected as an interleaved pair 0.56 s apart, so any window wide
enough for context contains both and a single window label would be ambiguous. Each frame is
classified from the 32 frames preceding it.

**Q: Does it run in real time?**
The demo runs a genuine forward pass per tick on CPU in a few milliseconds per frame. It is not
engineered as an embedded deployment, and we don't claim that.

**Q: What would you do next?**
Validate across vehicles and drive cycles, since a single capture cannot show generalisation; and
strengthen the residual representation for tampering, which is where the headroom is.

**Q: How can your model detect an attack it was never trained on? Isn't that impossible?**
It is impossible for the *classifier*, and we do not claim otherwise — with a 3-way output layer
it can only ever emit one of three labels. The detection comes from a second channel. The Digital
Twin is fitted on normal traffic only, so the residual |actual − expected| measures deviation
from normal, and that quantity is defined for any frame at all. The gate thresholds it at a level
set on held-out *normal* frames. There is no attack signature in it to have memorised.

**Q: Then couldn't the baselines just do the same thing?**
Not without a Digital Twin — which is the contribution. A CNN or BiLSTM over raw CAN traffic has
no reference for what the bus should be doing, only for what the attacks it was shown looked
like. You could bolt an anomaly detector onto a baseline, and that would be a different system;
the twin plus per-ECU residual fusion *is* the anomaly detector being proposed here.

**Q: How do I know the unknown class was really held out?**
`train_zeroday.py` removes it from the label space entirely (the heads are 3-way), drops every
training frame whose context window contains one, refits the twin on normal frames whose per-ECU
history contains none, and calibrates the gate on normal frames only. It prints the frame counts
it dropped and the app shows them. `check_demo.py` asserts the class is absent from the label
space and present in the test split.

**Q: Isn't a 100% detection rate just an alarm that fires on everything?**
That is the right question, and it is why every catch count in this project is shown next to a
false-alarm count on normal traffic — on the live cards, in the Results cost table, and in the
ROC sweep. A gate that alarmed on everything would show a false-alarm rate near 100%.

**Q: Does it work for every attack class?**
No. Hold out steering tampering instead and the proposed model does little better than the
baselines, because the forged steering payloads sit inside the range the twin already treats as
normal for that ID (§7d). The app reports that case as a negative result rather than hiding it.
The method needs an attack that actually moves payload bytes outside learnt-normal behaviour.

---

## 9. Project structure

```
TwinShield-Fusion/
├── app.py                      # entry point + top navigation
├── train.py                    # closed-set training pipeline
├── train_zeroday.py            # leave-one-attack-out (unknown attack) experiment
├── compute_shap.py             # SHAP attributions (run after train.py)
├── check_demo.py               # pre-demo readiness check
├── pages/
│   ├── 01_Overview.py          # pipeline, dataset, headline result
│   ├── 02_Live_Monitor.py      # four models side by side, two test cases
│   └── 03_Results.py           # zero-day + Digital Twin + closed set + audit
├── models/
│   ├── cnn.py                  # CNN baseline
│   ├── bilstm.py               # BiLSTM baseline
│   ├── cnn_bilstm.py           # CNN-BiLSTM backbone + detector
│   ├── digital_twin.py         # GRU Digital Twin + residual computation
│   ├── residual_fusion.py      # adaptive multi-ECU attention + TwinShield
│   └── anomaly_gate.py         # residual anomaly gate (the unknown-attack channel)
├── preprocessing/
│   ├── loader.py               # schema-adaptive GEM-CAN loader
│   ├── preprocessing.py        # frame features, with/without CAN ID
│   ├── windowing.py            # context windows, block split, ECU streams
│   └── prepare.py              # cached end-to-end preparation
├── explainability/
│   └── shap_explainer.py
├── evaluation/
│   ├── metrics.py
│   └── audit.py                # leakage / shortcut / duplicate audit
├── ui/                         # theme, shared components, cached data access
├── data/                       # GEM-CAN CSVs (not in version control)
├── saved_models/               # checkpoints, caches, zeroday__* artifacts
├── results/                    # results.json, zero_day.json, shap.npz, train_log.txt
├── requirements.txt
└── README.md
```

**Three pages, not four.** The Digital Twin used to have a page of its own; it now lives inside
Results, because the question a reviewer actually asks about the twin ("does the residual
separate this attack?") is a results question, and the screen the demo needs is the one where
all four models can be compared at once.

`ui/` is an addition to the originally sketched layout: the pages share a theme, a set of
card/pipeline components, and cached model access, and duplicating those across pages would have
been worse.

---

## 10. Method summary

1. **Features.** Per frame: 8 payload bytes (scaled), DLC, log inter-frame gap. The `with_id` set
   adds the same-ID gap and a one-hot arbitration ID; `no_id` omits both, since grouping by ID to
   compute a same-ID gap is itself ID information.
2. **Digital Twin.** GRU (96 hidden) with a 16-dim CAN-ID embedding, fed the last 8 payloads on
   that ID, predicting the next payload. Trained on **normal training frames only**, MSE loss.
3. **Residual.** `Residual_t = |Actual_t − Predicted_t|`, per byte → 8 dims per frame.
4. **Adaptive fusion.** Per-ECU mean residual within the window → additive attention across
   streams → fused vector. Absent streams are masked.
5. **Detector.** CNN-BiLSTM over the 32-frame window, concatenated with the fused residual vector,
   class-weighted cross-entropy for the 152:1 imbalance.
6. **Anomaly gate.** The channel that does not depend on known attacks, from the same residual
   through two views:

   - *frame view* — `z_frame = max over bytes b of (r[t,b] − μ[i,b]) / σ[i,b]`, where `i` is the
     frame's arbitration ID. The sharp channel: a forged byte far out of range shows up at once
     and is not averaged away by surrounding traffic.
   - *ECU view* — `z_ecu = max over streams present of (mean residual of the stream in the
     window − μ_s) / σ_s`. The sustained channel, over the multi-ECU structure.

   `score = max(z_frame, z_ecu)`, alarm when it exceeds the threshold. μ and σ come from normal
   **training** frames; the threshold is the (1 − target FPR) quantile over normal **validation**
   frames. No attack frame of any class is involved, which is the whole basis for calling the
   gate attack-agnostic. Standardising per ID and per byte matters: some ECUs and bytes (rolling
   counters, noisy sensors) sit at a high baseline residual, and one global threshold would fire
   on them constantly.
7. **Explainability.** `shap.GradientExplainer` over both the window and the ECU-stream inputs.

---

## 11. Licence and citation

Code: for academic use as part of a final-year research project.
Dataset: GEM-CAN, CC-BY-NC-3.0 — cite the Zenodo record (10.5281/zenodo.17834776) and the
accompanying *Data in Brief* article if you publish results.
