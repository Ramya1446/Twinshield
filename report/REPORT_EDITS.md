# Report edits — ready to paste

Seven changes. Items 1–6 are text; item 7 adds three figures.

Everything below is written to drop straight into the Word file. Where a number
has to come from your own run it is marked `[ACC]` and the command to read it is
given.

---

## 1. Title

Replace the title in **three places**: the title page, the DECLARATION paragraph
and the CERTIFICATE paragraph. The running title in the header, if you have one,
makes a fourth.

**Old**

> TwinShield-Fusion: A Digital Twin-Assisted Adaptive Multi-ECU Residual Fusion
> Framework for **Detecting Unseen Attacks on** Controller Area Networks

**New**

> TwinShield-Fusion: A Digital Twin-Assisted Adaptive Multi-ECU Residual Fusion
> Framework for **Intrusion Detection on Electric Vehicle** Controller Area Networks

On the title page it is set in capitals across four lines:

```
TWINSHIELD-FUSION: A DIGITAL TWIN-ASSISTED
ADAPTIVE MULTI-ECU RESIDUAL FUSION FRAMEWORK
FOR INTRUSION DETECTION ON ELECTRIC VEHICLE
CONTROLLER AREA NETWORKS
```

Two alternatives, if your guide prefers a shorter title:

- *TwinShield-Fusion: Digital Twin-Assisted Residual Fusion for Intrusion
  Detection on Electric Vehicle CAN Buses*
- *TwinShield-Fusion: An Adaptive Multi-ECU Residual Framework for Electric
  Vehicle In-Vehicle Network Security*

---

## 2 and 3. Abstract

Replace the whole abstract. This version opens on the electric vehicle and on
what intrusion detection is for, and it adds the train–validation–test split and
the accuracy alongside macro-F1.

> An electric vehicle carries out braking, steering and propulsion as digital
> commands exchanged between dozens of electronic control units over the
> Controller Area Network (CAN). Drive-by-wire platforms place safety-critical
> actuation on that bus, yet a CAN frame carries no sender identity, no message
> authentication code and no encryption, so any node that reaches the bus can
> issue a command indistinguishable from a genuine one. Replacing the protocol
> would mean replacing every control unit on every vehicle already on the road.
> The practical defence is therefore an intrusion detection system: a passive
> monitor that observes bus traffic and raises an alarm when it departs from
> expected behaviour.
>
> Learning-based intrusion detectors for CAN are almost always trained and
> evaluated on a fixed catalogue of attacks, and are then reported with
> closed-set metrics that assume every attack seen at test time was also present
> during training. A vehicle in service meets attacks that were in nobody's
> catalogue, and a supervised classifier has no output neuron for them.
>
> This project presents TwinShield-Fusion, an intrusion detection framework that
> pairs a conventional supervised detector with a second decision channel that
> never sees an attack during fitting. A gated recurrent unit digital twin is
> trained on normal traffic only and predicts the next payload on each
> arbitration identifier. The residual between prediction and observation is
> standardised per identifier and per payload byte, and an alarm threshold is
> calibrated on held-out normal frames at a target false-positive rate. Residuals
> are additionally pooled into per-ECU streams and combined by additive attention
> before being concatenated with a CNN-BiLSTM embedding of the raw context
> window.
>
> The system is evaluated on GEM-CAN, 143,444 real frames captured from a GEM e6
> autonomous electric vehicle on a closed track. Frames are split block-wise into
> 100,352 training, 21,504 validation and 21,588 test frames, with whole blocks
> of 256 consecutive frames assigned to one split at a time so that near-duplicate
> traffic cannot appear on both sides of the division. On the conventional
> closed-set benchmark TwinShield-Fusion reaches [ACC] per cent accuracy and
> 0.9979 macro-F1, and all four architectures compared fall within 0.0031
> macro-F1 of one another, so that benchmark does not discriminate between them.
> Under a leave-one-attack-out protocol, in which one attack class is removed
> from the label space, from every training context window, from the digital
> twin's per-ECU history and from the gate's calibration set, the same four
> models separate sharply: the CNN, BiLSTM and CNN-BiLSTM baselines flag 0.84%,
> 0.00% and 25.21% of the 119 held-out frames of the quarantined class, while
> TwinShield-Fusion flags 100.00% at a false-positive rate of 2.26% on normal
> traffic.
>
> The limits of the mechanism are reported with the same prominence as its
> successes. Brake tampering raises the twin's mean residual from 2.58 to 23.02
> bytes on the attacked identifier and is detected; steering tampering leaves it
> at 5.28 bytes against 6.39 on normal traffic, below the normal level, and a
> score defined as a positive deviation cannot fire on it. Detection therefore
> does not require prior knowledge of an attack, but it does require the attack
> to be anomalous in the payload. These are different conditions, and the project
> quantifies that boundary rather than asserting past it.

**Fill in `[ACC]`.** Run this in your project folder:

```
python -c "import json; d=json.load(open('results/results.json')); print(d['results']['with_id']['TwinShield-Fusion']['metrics'])"
```

Use the `Accuracy` value, as a percentage to one decimal place.

A caution worth knowing before your viva. Section 6.2 and Section 6.3 of your own
report argue that accuracy is a poor headline metric on this dataset, because a
lookup table on the arbitration identifier alone already reaches 98.9 per cent.
Quoting accuracy next to macro-F1, as above, is fine. Quoting it alone would
contradict your Chapter 6, and an examiner who has read that chapter may well ask.

---

## 4. Keywords

Seven keywords, naming the models, with open-set recognition and explainable AI
removed.

**Old**

> Keywords: Controller Area Network, intrusion detection, digital twin, anomaly
> detection, zero-day attacks, automotive security, open-set recognition,
> explainable AI

**New**

> Keywords: Controller Area Network, electric vehicle security, intrusion
> detection, digital twin, GRU, CNN-BiLSTM, anomaly detection

---

## 5. Introduction

Section 1.1 already covers the protocol and the demonstrated attacks. Three
insertions give it the electric vehicle, intrusion detection and the challenges.

### 5a. Insert after the first paragraph of 1.1 (the one ending "the message with the lowest identifier always wins")

> The electric vehicle has made this inheritance more consequential. An internal
> combustion vehicle retains mechanical linkages between the driver and the
> wheels; an electric platform, and an autonomous one in particular, replaces
> them with drive-by-wire actuation in which braking, steering and propulsion are
> issued as CAN frames. Battery management, regenerative braking and motor
> torque control add further high-rate traffic to the same bus. The result is
> that more safety-critical function is expressed as bus messages, on a vehicle
> whose connectivity gives an attacker more ways to reach that bus.

### 5b. Replace the final paragraph of 1.1 (the one beginning "Replacing CAN with an authenticated protocol")

> Replacing CAN with an authenticated protocol would require replacing every ECU
> on every vehicle, which is not economically realistic for a fleet already on
> the road. The practical response adopted by both industry and the research
> community has therefore been to add an intrusion detection system. An IDS is a
> passive monitor: it reads every frame on the bus, forms a judgement about
> whether traffic looks the way it should, and raises an alarm when it does not.
> It changes no message and blocks nothing, which is what makes it deployable on
> an existing vehicle, and it is the only defence that can be retrofitted without
> touching the protocol. Intrusion detection for CAN divides into two families:
> signature-based detectors, which look for patterns known to be malicious, and
> anomaly-based detectors, which model what is normal and report departures from
> it. This project is concerned with the question of what "should" means in that
> judgement, and in particular with what happens when an attack does not resemble
> anything the monitor was taught to recognise.

### 5c. Insert as a new final subsection of 1.1, headed "Challenges"

> **Challenges.** Four properties of the problem make intrusion detection on a
> vehicle bus harder than it first appears.
>
> The traffic is extremely repetitive. ECUs transmit fixed status payloads at
> fixed intervals, so a large share of frames is byte-identical to another frame
> elsewhere in the capture. A model can score well by memorising the repetition
> rather than learning the behaviour, and a careless train–test split will
> flatter it further by placing near-duplicate frames on both sides.
>
> The classes are severely imbalanced. A flooding attack produces tens of
> thousands of frames, while a tampering attack produces a few hundred. Accuracy
> is dominated by the majority classes and reports almost nothing about the rare
> ones, which are the attacks that matter.
>
> Some attacks are trivially separable and some are not. A denial-of-service
> flood announces itself through an arbitration identifier no legitimate ECU uses,
> so a lookup table detects it. Tampering attacks forge payloads on identifiers
> that carry genuine traffic throughout the capture, so the identifier carries no
> information and the decision must come from the payload or the timing.
>
> The attack set is open. A detector trained on a catalogue of attacks must still
> behave sensibly when it meets one that is absent from that catalogue, and a
> closed-set evaluation never asks it to.

---

## 6. Section 1.5, Scope of the Project

Replace the two bold run-in headings and re-angle the first paragraph so it reads
as what the work is for rather than as a scope boundary.

**Old:** "In scope." and "Out of scope."

**New:** "Applications of the work." and "Boundaries of the study."

### Replacement text for 1.5

> The scope of this project is deliberately bounded so that each component can be
> validated rather than merely demonstrated.
>
> **Applications of the work.** The system is built and evaluated as a passive
> bus monitor for an electric, drive-by-wire vehicle, and this is the setting in
> which it applies: an in-vehicle intrusion detection unit on a platform where
> braking and steering are issued as CAN messages. It operates on the GEM-CAN
> capture, comprising 143,444 real frames recorded from a GEM e6 autonomous
> electric vehicle on a closed track with a 250 kbit/s bus. Four classes are
> supported, being exactly those the dataset contains: normal traffic,
> denial-of-service flooding, brake tampering and steering tampering. Detection
> is performed at frame level using a 32-frame context window. Every model is
> trained twice, once with the arbitration identifier available as an input and
> once with it removed, so that the identifier shortcut identified in the audit
> can be eliminated rather than merely acknowledged. The leave-one-attack-out
> experiment is run with brake tampering as the quarantined class. Beyond the
> vehicle itself, the same construction applies wherever a fleet operator or a
> security tester needs a detector that reports an attack it was never trained
> on, and the interactive application built in Chapter 5 is intended for exactly
> that kind of inspection.
>
> **Boundaries of the study.** The project stops short of real-time deployment on
> embedded automotive hardware, and makes no claim about line-rate operation on a
> vehicle ECU. Attack prevention, message authentication and key management lie
> outside it, since all three would require protocol changes. Validation is
> confined to a single capture, so claims across multiple vehicles, drive cycles
> or attack families are left for future work. Replay attacks, in which a genuine
> past message is retransmitted at the wrong moment, are discussed as a known
> limitation; the dataset contains none, so they are not evaluated.

---

## 7. Confusion matrix, ROC curve and precision–recall curve

Generate them with `make_eval_figs.py`, which reads the test-split probabilities
your training run already saved:

```
cd report
python make_eval_figs.py
```

It writes three files into `figures/` and prints every value it plots:

| File | What it shows |
|---|---|
| `fig6_6_confusion.png` | 4x4 confusion matrix for TwinShield-Fusion, counts and row shares |
| `fig6_7_roc.png` | One-vs-rest ROC, one panel per class, AUC in each panel title |
| `fig6_8_pr.png` | Precision–recall, one panel per class, average precision and the no-skill baseline |

### Where they go

They belong in **Section 6.5, Closed-set comparison**, immediately after Table 6.2
and Figure 6.1, because they describe the closed-set behaviour of the proposed
model in more detail than a single macro-F1 figure can.

That placement means the three new figures become **6.2, 6.3 and 6.4**, and
everything after them shifts by three: the old 6.2 becomes 6.5, the old 6.3
becomes 6.6, and so on up to the old 6.8 becoming 6.11. The List of Figures and
every in-text reference has to move with them.

If you would rather avoid that renumbering, place them at the end of Chapter 6
instead, as a new section 6.12 titled **Classification Diagnostics**, with the
existing 6.12 User Interface becoming 6.13 and so on. The figures then take the
next free numbers, 6.9, 6.10 and 6.11, and only the section numbers shift.

Send me the `.docx` and I will apply whichever you prefer, renumbering included.

### Caption and prose to accompany them

> **Figure 6.2: Confusion matrix for TwinShield-Fusion on the held-out test split**
>
> Figure 6.2 gives the full confusion structure behind the macro-F1 figure in
> Table 6.2. Each cell carries the frame count and, beneath it, that count as a
> share of the true class. The diagonal dominance on normal traffic and on DoS is
> expected, since both are well represented and DoS is separable by its
> arbitration identifier alone. The two tampering classes are the informative
> rows: with 119 and 118 test frames respectively, a handful of misclassified
> frames moves their per-class F1 noticeably, which is why macro-F1 rather than
> accuracy is used throughout this chapter.
>
> **Figure 6.3: One-vs-rest ROC curves with areas under the curve**
>
> Figure 6.3 plots the receiver operating characteristic for each class in turn,
> treating that class as positive and the remaining three as negative. Each panel
> shows the trade-off between true-positive rate and false-positive rate as the
> decision threshold on that class's probability is swept from one to zero; the
> dashed diagonal is the line a classifier with no discriminative power would
> follow. The area under each curve is reported in its panel title, and the
> support is given at the lower right, since an area computed over 118 positive
> frames is a less stable estimate than one computed over 15,063.
>
> **Figure 6.4: One-vs-rest precision–recall curves with average precision**
>
> Figure 6.4 reports the same four decisions as precision against recall. On a
> split this imbalanced the precision–recall curve is the more honest of the two
> views, because the false-positive rate underlying a ROC curve is diluted by a
> large negative class, whereas precision is not. The dashed line in each panel
> is the no-skill baseline, equal to that class's share of the test split: 0.698
> for normal traffic, and under 0.01 for each tampering class. A curve that sits
> far above its own baseline is doing real work; the gap, rather than the
> absolute height, is what should be read.

---

## Order of work

1. Run `make_eval_figs.py` and note the printed AUC and average-precision values.
2. Fill `[ACC]` in the abstract.
3. Paste items 1 to 6.
4. Insert the three figures and renumber.
5. In Word: Ctrl+A, then F9, and choose "Update entire table" so the contents
   list, list of figures and list of tables pick up the new page numbers.
