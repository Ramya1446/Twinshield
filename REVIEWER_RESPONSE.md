# Response to the reviewer comments

Every comment, what was done about it, and what you still have to run.

Three groups: **done in the paper**, **run these**, **yours to fill in**.

---

## 1. Done in the paper already

These are fixed in the `.docx` I sent. Nothing for you to do.

| Comment | What changed |
|---|---|
| Ablation called "monotonic" but Table III shows a drop | The paragraph no longer claims monotonicity. It reads the table as it stands: with the identifier included the scalar twin lowers macro-F1 from 0.9979 to 0.9958 and only the per-byte residual recovers it; with the identifier excluded the ladder does rise at every step. |
| "The residual does tangible work" on differences of ~0.001 | The claim is withdrawn. The text now states that macro-F1 weights the 119- and 118-frame tampering classes as heavily as the 15,063-frame normal class, so one frame moves macro-F1 by about 0.002 and a 0.0065 difference is of the order of three frames. It reports a direction, not a resolved quantity. |
| Abstract says four within 0.004, Table II lists five, ID-excluded spread is ~0.008 | The conclusion quoted 0.004 unqualified; it now reads 0.0031, names the four architectures, and names the variant. Section VI already carried both spans (0.0031 for four, 0.0041 for five). |
| Twin MAE of 10.52 "bytes" inconsistent with payloads scaled to [0,1] | Now given both ways: 0.041 in the scaled units the model optimises, equivalently 10.52 on the 0–255 byte scale. Both are what `train.py` computes, as `MAE_normalised` and `MAE`. |
| Figures out of order (Fig. 4 before Fig. 3) | Numbers swapped, in both captions and both in-text references. |
| Diagram to be enhanced | Figure 1 redrawn: both branches shown stage by stage, each box keeping the Section IV sub-section it belongs to, sized to the IEEE column. |
| 2.26% is too high for a real vehicle | Section VI-C now states it per thousand frames and as a rate per second, and describes the operating point as a research setting rather than a deployable alarm. `bus_rate.py` gives the actual number — see below. |

### Comments that did not apply to the file you sent

I checked each against the document. If your reviewer saw these, they were reading an
older or paraphrased copy — worth confirming which file they had.

- "Equations 3–5 are missing" — all three are present and numbered.
- "Section IV-D garbled: *One common global threshold keeps firing on them constantly*" — the file reads "A single global threshold fires on them continuously."
- "Typos: ortcut, 12.0 ms per image" — "shortcut" is intact; the text says "per frame".
- "We always pre-trained whatever we were auditing for shortcuts" — the file reads "Before training anything we audited the dataset for shortcuts."
- "Müter, Taylor, Hanselmann and Cho unnumbered; refs [14]–[17] uncited" — all four are numbered [14], [16], [17], [18], and all 26 references are cited.

---

## 2. Run these

Four new scripts, all in the project root. Each answers one reviewer demand and
writes its own JSON into `results/`.

Install once:

```
pip install matplotlib
```

### 2.1 Real baselines — `extra_baselines.py`

> *Add real baselines: autoencoder, one-class model, open-set scores, the twin
> residual alone, and a simple per-ID range check.*

```
python extra_baselines.py
```

Needs `train_zeroday.py` to have run first; it reuses that run's twin, residuals,
streams, gate and detectors rather than refitting anything, so the numbers land
beside the ones already in `results/zero_day.json`. Minutes, not hours.

It adds six detectors under the identical quarantine, each calibrated the same
way the gate is — on held-out normal validation frames at a common target
false-positive rate:

| Baseline | What it is |
|---|---|
| Per-ID range check | Flags a payload byte never seen on that identifier in normal training traffic. No learning, no threshold. |
| Isolation forest | A standard one-class detector on frame features. |
| Autoencoder | Reconstruction error on frame features, fitted on normal only. |
| Twin residual alone | The proposed gate with the supervised head removed. |
| MSP, Energy | Open-set scores read off each supervised baseline. |

**Be ready for one outcome.** The per-ID range check may well detect brake
tampering, because the forged bytes sit far outside the range that identifier
normally carries — that is the same property the twin residual keys on. If it
does, say so plainly in the paper. The defensible claim is then about *steering*
tampering, false-alarm cost and generality, not about brake tampering alone. A
reviewer who suspects a trivial baseline would have worked will respect the
paper that tested it and reported the result.

### 2.2 Every class, several seeds, confidence intervals — `loao_sweep.py`

> *Run the leave-one-attack-out test on every attack class, with several random
> seeds and confidence intervals.*

```
python loao_sweep.py --dry-run     # see the plan
python loao_sweep.py               # 3 classes x 5 seeds = 15 runs
```

Each run is a separate call to `train_zeroday.py`, so the sweep cannot drift from
the experiment the paper describes. Runs are archived to
`results/loao_runs/` and skipped if already present, so you can stop and restart
it. It reports mean detection and a 95% Student-t interval per model per class,
clipped to [0, 100] because rates are bounded.

**This is the long one.** One run is a digital twin plus four detectors; fifteen
runs is that fifteen times. Start it before you sleep. If time is short:

```
python loao_sweep.py --seeds 1 2 3            # 9 runs
python loao_sweep.py --classes "Brake Tampering" "Steering Tampering" --seeds 1 2 3
```

Three seeds is enough to put an interval on the number. One seed is not.

### 2.3 False alarms per second — `bus_rate.py`

> *Report false alarms per second, since 2.26% is too high for a real vehicle.*

```
python bus_rate.py
```

Seconds to run. It measures the frame rate from the capture's own timestamps —
mean, median and busiest five per cent — and converts every false-positive rate
in `results/zero_day.json` into alarms per second. Put the resulting number into
Section VI-C where the text currently says the rate is `0.0226f`.

### 2.4 A second dataset — `convert_dataset.py`

> *Test on at least one more dataset (ROAD, SynCAN, or Car-Hacking).*

```
python convert_dataset.py --preset car-hacking --src <path-to-Car-Hacking> --out data/CarHacking_as_GEM
GEMCAN_DIR=data/CarHacking_as_GEM python train.py
GEMCAN_DIR=data/CarHacking_as_GEM python train_zeroday.py --unknown "DoS"
```

It converts a third-party dataset into the two CSVs this project reads, then
verifies the result by loading it back through the project's own loader.

**One thing you must not get wrong.** This project's label space is fixed at four
slots: Normal, DoS, Brake Tampering, Steering Tampering. Car-Hacking's classes are
DoS, Fuzzy, gear spoofing and RPM spoofing. The converter puts each foreign class
into a slot and records the assignment in `conversion.json`, printing it on every
run. **Every figure and table from converted data has to be relabelled using that
file.** Writing "brake tampering" about a Fuzzy run would be a false statement
about an experiment, and it is the kind of error that ends a submission.

Car-Hacking is the easiest of the three to obtain and the preset is built in.
ROAD and SynCAN need `--preset generic` with their column names.

---

## 3. Yours to fill in

| Item | Where | How |
|---|---|---|
| Four `[verify]` cells | Table IV, known-class macro-F1 | `results/zero_day.json` → `models` → *name* → `known_class_metrics` → `Macro F1` |
| Author block | First page | Names, affiliations, email |
| IEEE header | First page | Date of publication, DOI — the publisher fills these |

---

## 4. Suggested order

1. `pip install matplotlib`
2. `python bus_rate.py` — seconds. Gives you one number for Section VI-C.
3. `python extra_baselines.py` — minutes. Gives you a new table.
4. `python loao_sweep.py --seeds 1 2 3` — hours. Start it overnight.
5. `python convert_dataset.py ...` then train on it — longest, do last.

Steps 2 and 3 alone answer two of the five experimental demands and cost you an
evening. If the deadline only allows those, say in the limitations section which
of the remaining experiments were not run, which the paper now does explicitly.

---

## 5. What I could not do, and how far the code was tested

The five experimental demands need your data and your machine. The numbers have
to come from your capture, and I will not put numbers in a paper that I did not
measure.

How far each script was actually exercised, so you know what to watch for:

| Script | Tested how |
|---|---|
| `bus_rate.py` | Run end to end on a synthetic capture. Frame-rate arithmetic checked against the generator's known inter-arrival distribution. |
| `convert_dataset.py` | Run end to end on a synthetic Car-Hacking-shaped source, and the output loaded back through `preprocessing/loader.py` to confirm the labels survive. |
| `extra_baselines.py` | Scoring functions unit-tested: the range check recovered 100% of injected out-of-range frames at a 1% false-alarm rate, the isolation forest 100% at 0.7%, the calibration quantile matched the analytic value, and MSP and energy ordered confident and uniform predictions correctly. Then run end to end against a full training run on a synthetic capture. |
| `loao_sweep.py` | Aggregation, resumption and the confidence interval checked against archived runs with known means. The sweep driver calls `train_zeroday.py` unchanged. |

A synthetic capture exercises the code paths. It cannot tell you whether a
baseline beats your method on your data — only your run can do that.

---

## 6. Draft text for the new results, ready to fill

When `extra_baselines.py` and `loao_sweep.py` have run, these go into the paper.
Every number is a slot marked `__` — fill them from the scripts' own output and
change nothing else. If a sentence turns out to be false on your data, change the
sentence, not the number.

### 6.1 New subsection for Section VI, after VI-B

> **C. UNSUPERVISED AND OPEN-SET BASELINES**
>
> The comparison in Section VI-B is between four supervised architectures, only
> one of which has a second channel. That is not the comparison a reader should
> accept, because it leaves open whether the gate earns its place or whether any
> unsupervised detector would do. Table VI answers it. Every row is fitted on
> normal traffic from the same quarantined training split, and every row with a
> threshold is calibrated the same way the gate is: on held-out normal validation
> frames at a one per cent target false-positive rate, with no attack frame of any
> class taking part.
>
> *[Table VI here: Detector / Detected / False alarms, from extra_baselines.py]*
>
> The per-identifier range check is the row that matters most, because it is the
> cheapest thing that could work: it flags a payload byte never seen on that
> identifier in training, and it has no parameters at all. It detects __% of the
> quarantined class at __% false alarms. [If it detects most of them:] On this
> attack, that simple test is competitive with the digital twin, and the twin's
> advantage is therefore not that it detects brake tampering but that it does so
> __. [If it does not:] The forged bytes evidently fall inside the per-identifier
> range often enough that a static bound misses __% of them, which is what the
> learnt per-byte distribution adds.
>
> The isolation forest and the autoencoder operate on the same frame features as
> the supervised baselines and detect __% and __% respectively. The twin residual
> acting alone, with the supervised head removed entirely, detects __%, which is
> consistent with the decomposition in Section VI-B attributing the whole of the
> detection to the gate.
>
> The two open-set rules read off the supervised baselines detect between __% and
> __%. A softmax trained on k classes assigns an out-of-set attack to whichever of
> them fits least badly, and thresholding its confidence does not recover what the
> label space never contained.

### 6.2 Replacement for the Table V paragraph, once the sweep has run

> Table V reports one quarantined class at one seed. Table VII repeats the
> experiment for every attack class the dataset supports, across __ seeds, and
> reports the mean with a 95 per cent interval. [Then one of:] The ordering in
> Table V holds across seeds and classes, with intervals that do not overlap
> between TwinShield-Fusion and the strongest baseline. / The ordering holds for
> __ but not for __, where the intervals overlap and no claim of separation is
> supportable.

### 6.3 Sentence for Section VI-C, once bus_rate.py has run

> At the __ frames per second this capture sustains on average, a 2.26 per cent
> false-positive rate is __ alarms per second.

### 6.4 If the second dataset runs

> **D. A SECOND CAPTURE**
>
> [Dataset name] was converted to the same frame schema and the same protocol
> applied, with __ quarantined. [Result.] The classes in that dataset are not the
> classes in GEM-CAN, and the mapping used is recorded with the converted data;
> the slot names in our code are ours, and the attacks they stand for are
> [dataset]'s.

### A note on how to use these

Write the result you get, not the result you wanted. Two of these drafts have an
if/else because I do not know which way your data will fall, and the paper is
stronger for choosing the branch honestly than for quietly picking the flattering
one. A reviewer who already suspects the baselines were skipped will read a
suspiciously clean table as confirmation.
