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

## 5. What I could not do

The five experimental demands need your data and your machine. I wrote and tested
the code against a synthetic capture built to GEM-CAN's documented structure —
every code path runs — but the numbers have to come from your capture, and I will
not put numbers in a paper that I did not measure.
