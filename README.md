# 🛡️ TwinShield-Fusion

**A Digital Twin-Assisted Adaptive Multi-ECU Residual Fusion Framework for Explainable Intrusion Detection in CAN Networks**

Final-year CSE research project — CAN-bus intrusion detection on **real** electric-vehicle traffic.

```
Real CAN Traffic → Digital Twin → Residual → Adaptive Multi-ECU Fusion → CNN-BiLSTM → Detection → SHAP
```

Every number the application displays — accuracy, F1, confidence, residuals, attention weights,
SHAP values, and the CAN frames themselves — is computed from the real GEM-CAN dataset or from a
trained model checkpoint. Nothing is simulated, hard-coded, or fabricated.

---

## 1. Quick start

```bash
pip install -r requirements.txt
```

```bash
python train.py
```

```bash
python compute_shap.py
```

```bash
python check_demo.py
```

```bash
streamlit run app.py
```

Train once before the presentation. The app then loads saved models and starts instantly —
judges never wait for training. `check_demo.py` verifies every artifact and runs one real
inference, so a missing file surfaces before you are standing in front of the panel.

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

### Saving and loading

Saving is automatic. The app loads through `ui/data.py`:

```python
from ui.data import get_detector, get_digital_twin, infer
model = get_detector("with_id")       # TwinShield-Fusion
twin  = get_digital_twin()            # GRU Digital Twin
proba, attention = infer(frame_indices)   # real forward pass
```

Checkpoints are named `{variant}__{arch}__{feature_mode}.pt`, e.g.
`with_id__twinshield__fusion.pt`. Delete `saved_models/prepared_dataset.pkl` to force the
dataset cache to rebuild.

---

## 4. Demo instructions

**Route:** Overview → *START LIVE DEMO* → Live Monitor → Digital Twin → Results. About 3 minutes.

1. **Overview** — pipeline, dataset card, feature cards.
2. **Live Monitor** — pick an *Attack Scenario*, press **▶ Replay Attack**. Real frames stream in,
   the status banner flips from 🟢 to 🔴 when the model's own prediction changes, the residual
   trace and ECU attention update from live inference. **⏸ Pause** to talk over a frozen screen.
3. **Digital Twin** — Expected vs Actual for a chosen CAN ID and payload byte, plus MAE/RMSE/MSE.
4. **Results** — model cards, comparison table, per-class F1, ablation ladder, SHAP, audit,
   auto-generated result statement.

**Before you present:** run `python train.py` and `python compute_shap.py`, then open the app
once so the caches are warm. Have `results/results.json` handy as a backup.

---

## 5. What to say on each page

**Overview (~20 s)**
> "This detects cyberattacks on the CAN bus of an electric vehicle. We train a Digital Twin on
> normal traffic so it knows what the car should be saying. Anything that deviates produces a
> residual, we fuse residuals across ECUs, classify with a CNN-BiLSTM, and explain the decision
> with SHAP. The data is real — 143,000 frames from a GEM e6."

**Live Monitor (~60 s)**
> "These are real CAN frames replayed in recorded order. For each frame the Digital Twin predicts
> the expected payload, we take the residual, attention decides which ECU stream matters, and the
> CNN-BiLSTM classifies it. Watch the residual as the attack starts — the banner turns red only
> because the model's own output changed."

**Digital Twin (~40 s)**
> "The twin is a GRU trained only on normal traffic. The dotted line is what it expects, the solid
> line is what actually happened. During tampering they separate, and that gap is the residual we
> feed the detector."

**Results (~60 s)**
> "All models share one held-out test split. Here's the ablation: backbone, plus Digital Twin,
> plus residual, plus adaptive fusion. And here's the part we want to be upfront about — CAN ID
> alone already gets 98.9% accuracy on this dataset, because every DoS frame uses ID zero. So we
> report macro-F1, not accuracy, and we re-ran everything with the ID removed."

---

## 6. Measured results

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

---

## 9. Project structure

```
TwinShield-Fusion/
├── app.py                      # entry point + top navigation
├── train.py                    # full training pipeline
├── compute_shap.py             # SHAP attributions (run after train.py)
├── check_demo.py               # pre-demo readiness check
├── pages/
│   ├── 01_Overview.py
│   ├── 02_Live_Monitor.py
│   ├── 03_Digital_Twin.py
│   └── 04_Results.py
├── models/
│   ├── cnn.py                  # CNN baseline
│   ├── bilstm.py               # BiLSTM baseline
│   ├── cnn_bilstm.py           # CNN-BiLSTM backbone + detector
│   ├── digital_twin.py         # GRU Digital Twin + residual computation
│   └── residual_fusion.py      # adaptive multi-ECU attention + TwinShield
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
├── saved_models/               # checkpoints + cached dataset
├── results/                    # results.json, shap.npz, train_log.txt
├── requirements.txt
└── README.md
```

`ui/` is an addition to the originally sketched layout: the four pages share a theme, a set of
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
6. **Explainability.** `shap.GradientExplainer` over both the window and the ECU-stream inputs.

---

## 11. Licence and citation

Code: for academic use as part of a final-year research project.
Dataset: GEM-CAN, CC-BY-NC-3.0 — cite the Zenodo record (10.5281/zenodo.17834776) and the
accompanying *Data in Brief* article if you publish results.
