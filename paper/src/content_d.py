# -*- coding: utf-8 -*-
"""Sections 7 (Results), 8 (Discussion) and 9 (Conclusion)."""

SEC7 = [
    ("sec", "VII.  RESULTS AND ANALYSIS"),
    ("sub", "A.  PERFORMANCE ON THE GEM-CAN TEST SET"),
    ("body",
     "Table IV gives closed-set macro-F1 for every model on the shared held-out test "
     "split, with and without the arbitration identifier as an input. With the "
     "identifier included, the four architectures span 0.0031 macro-F1 and all five "
     "rows span 0.0041. Neither number is evidence that one architecture is better "
     "than another; both are evidence that the benchmark has stopped measuring "
     "architecture. A lookup table on the identifier alone reaches 98.9% accuracy on "
     "this capture, so there is very little label information left for a sequence "
     "model to add."),
    ("tlabel", "TABLE IV"),
    ("tcap", "Closed-set macro-F1 on the shared GEM-CAN test split"),
    ("table", "GEM_CLOSED"),
    ("body",
     "Removing the identifier is more informative. There the digital-twin residual "
     "lifts the CNN-BiLSTM backbone from 0.9914 to 0.9979 macro-F1. That gain should "
     "be read against the support behind it: macro-F1 is weighted heavily by the two "
     "tampering classes, which contain 119 and 118 test frames, so a single frame "
     "changing class moves a tampering F1 by about 0.008 and macro-F1 by about 0.002. "
     "A difference of 0.0065 is therefore of the order of three frames. We report it "
     "as a consistent direction, not as a precisely resolved quantity."),
    ("body",
     "The result that separates the models is the open-set one. Table VI reports the "
     "leave-one-attack-out experiment with brake tampering quarantined from every "
     "training stage. The three supervised baselines raise an alarm on 0.84%, 0.00% "
     "and 25.21% of the 119 held-out frames; TwinShield-Fusion raises one on all 119. "
     "The known-class macro-F1 column confirms that the proposed model has not bought "
     "that detection by degrading on the classes it kept."),
    ("tlabel", "TABLE V"),
    ("tcap", "Ablation on the same CNN-BiLSTM backbone, GEM-CAN (macro-F1)"),
    ("table", "GEM_ABL"),
    ("tlabel", "TABLE VI"),
    ("tcap", "Leave-one-attack-out on GEM-CAN: brake tampering quarantined from every "
             "training stage"),
    ("table", "GEM_ZD"),
    ("fig2", None),
    ("fcap",
     "FIGURE 2.  Detection of a quarantined attack class on GEM-CAN. Brake tampering "
     "was removed from the label space, from every training context window, from the "
     "digital twin's per-ECU history and from the gate's calibration set. Detection "
     "rate is measured on its 119 held-out test frames; the false-positive rate is "
     "measured on 15,063 normal test frames."),
    ("body",
     "The credit belongs to one component and the decomposition is worth stating. "
     "TwinShield-Fusion's supervised head contributes none of that detection, because "
     "it has no label to assign; the anomaly gate contributes all of it. The residual "
     "is the generaliser and the classifier is the namer. The 25.21% achieved by "
     "CNN-BiLSTM is not recognition either: that model also has no output for brake "
     "tampering, and what it does is place some of those frames in steering tampering, "
     "the other tampering class it was trained on, which shares an identifier family "
     "and a timing signature. The alarm is raised by misattribution. The two weaker "
     "baselines cannot even do that."),
    ("sub", "B.  PERFORMANCE ON THE CICIoV2024 TEST SET"),
    ("body",
     "The second capture asks whether the result survives a change of vehicle, of attack family and of recording conditions. Table VII gives closed-set macro-F1. With the arbitration identifier available every model scores 1.0000, and that is not a result about models: on this capture the identifier alone is a perfect classifier, because not one of the attack identifiers appears anywhere in the 1,223,737 benign frames of the release. The informative column is therefore the one with the identifier removed, and there the digital twin lifts the CNN-BiLSTM backbone from 0.9992 to a perfect 1.0000, with per-class F1 of 1.0000 on every class. The twin also fits this vehicle far more closely than the GEM e6, reaching a mean absolute error of 1.88 bytes on 26,880 held-out normal frames against 10.52 on GEM-CAN."),
    ("tlabel", "TABLE VII"),
    ("tcap", "Closed-set macro-F1 on the CICIoV2024 test split"),
    ("table", "CIC_CLOSED"),
    ("body",
     "Table VIII repeats the leave-one-attack-out protocol, once for each of the three attack slots. TwinShield-Fusion raises an alarm on 100.00% of the held-out frames in all three runs, at a false-positive rate of 0.37% on normal traffic, while the three supervised baselines reach 0.00%, 35.71% and 0.16% at best. Known-class macro-F1 remains 1.0000 in every run, so the detection is not bought by degrading on the classes that were retained. The margin is wide rather than marginal: against a gate threshold of z = 4.18, the mean standardised residual of the held-out class is 109 for DoS, 104 for powertrain spoofing and 152 for steering spoofing, and the per-frame view carries the whole of each decision."),
    ("tlabel", "TABLE VIII"),
    ("tcap", "Leave-one-attack-out on CICIoV2024: each attack slot quarantined from "
             "every training stage in turn"),
    ("table", "CIC_ZD"),
    ("body",
     "Two qualifications belong with that table. The first is that the separation is "
     "complete over the whole threshold range, not only at the operating point: the "
     "area under the detection-against-false-alarm curve is 1.000 for all three "
     "classes, and 100% detection is still reached at the smallest false-alarm rate "
     "the sweep resolves. Fig. 6 plots it. That is a strong result, and it is also a "
     "warning sign."),
    ("body",
     "The second is the redundancy documented in Section III-C. CICIoV2024 contains "
     "3,588 distinct identifier-and-payload tuples in 1,408,219 frames, its attack "
     "classes are drawn from as few as two or three distinct tuples, and no attack "
     "identifier appears anywhere in the benign capture. A detector facing that "
     "capture is close to facing a lookup problem, and a perfect score on it should "
     "be read accordingly. We report these numbers because they are the honest "
     "context for the table above: the CICIoV2024 result confirms that the method "
     "transfers to a second vehicle and a second attack family, and it does not, on "
     "its own, establish detection capability at the level the raw figures suggest. "
     "The GEM-CAN experiment, where the tampering attacks ride on identifiers that "
     "also carry legitimate traffic, is the harder and more informative of the two."),
    ("body",
     "To put an uncertainty on those figures rather than relying on a single seeded "
     "run, the steering-spoofing quarantine was repeated with four random seeds. "
     "Detection is perfectly stable: 100.00% on every seed, with a standard deviation "
     "of zero, as are the known-class macro-F1 of 1.0000 and the area under the curve "
     "of 1.000. The false-alarm rate is not stable. Across the four seeds it takes "
     "the values 0.37%, 0.35%, 0.55% and 0.02%, a mean of 0.32% with a standard "
     "deviation of 0.22, so the operating point moves by more than an order of "
     "magnitude while the detection rate does not move at all."),
    ("body",
     "That asymmetry is informative and we report it rather than quoting the best "
     "seed. It follows from how the threshold is set: the gate\u2019s quantile is read "
     "from whichever normal validation frames the split assigns, and on a capture "
     "whose payload vocabulary is as small as this one a different draw of validation "
     "blocks shifts the tail of the normal score distribution appreciably. The "
     "detection rate is insensitive to that shift because, as Fig. 6 shows, the "
     "attack and normal score distributions do not overlap at all here. On a harder "
     "capture the same sensitivity would move the detection rate as well, which is "
     "the reason the false-alarm figures in this paper should be read as an operating "
     "region rather than as a single number."),
    ("sub", "C.  COMPARATIVE ANALYSIS ACROSS DATASETS"),
    ("body",
     "Table IX places the quarantined-class result from both vehicles side by side. Across two real captures and four held-out classes, TwinShield-Fusion raises an alarm on every held-out frame, while the best supervised baseline on any of them reaches 35.71% and is usually at or near zero. The false-alarm cost differs by nearly an order of magnitude between the captures, 2.26% on GEM-CAN against 0.37% on CICIoV2024, which tracks how well the twin fits each vehicle: a mean absolute error of 10.52 bytes on the GEM e6 against 1.88 on the Ford."),
    ("tlabel", "TABLE IX"),
    ("tcap", "Cross-dataset summary of the quarantined-class result"),
    ("table", "CROSS"),
    ("fig5", None),
    ("fcap",
     "FIGURE 3.  Why the gate transfers or does not. Each marker is one arbitration "
     "identifier under one attack class, placed at the ratio of its mean digital-twin "
     "residual under attack to its mean residual on normal traffic. Only an attack "
     "that drives the residual far outside its normal range is separable by a gate "
     "calibrated at a 1% false-alarm rate."),
    ("body",
     "A third capture supports the same reading. SynCAN [21], the synthetic benchmark "
     "released with CANet, consists entirely of signal-space manipulations that hold a "
     "signal at a constant valid value, drift it slowly, replay it or suppress it. "
     "Converted to this schema and run under the identical protocol, the gate detects "
     "at most 7.07% of a quarantined class there, and the median ratio of attack "
     "residual to normal residual across thirty identifier-and-class pairs is 1.04. A "
     "control in which each signal is packed into one byte rather than two makes the "
     "twin five times more accurate and leaves detection unchanged at 1.56%, which "
     "rules out the quantisation as the cause. The pattern across all three captures "
     "is the same, and it is a property of the attack rather than of the dataset."),
    ("sub", "D.  COMPARISON WITH BASELINE MODELS"),
    ("body",
     "Fig. 4 places the same four models under the two protocols. Under the closed-set "
     "protocol they are indistinguishable, separated by 0.0031 macro-F1. Under the "
     "leave-one-attack-out protocol on GEM-CAN they are separated by more than 74 "
     "percentage points. The comparison is the central argument of this paper: a "
     "benchmark on which four architectures agree to three decimal places is not "
     "measuring the property that matters, and changing the question rather than the "
     "model is what makes the difference visible."),
    ("fig3", None),
    ("fcap",
     "FIGURE 4.  The same four models under two protocols on GEM-CAN. Closed-set "
     "macro-F1 separates them by 0.0031; detection of a quarantined attack class "
     "separates them by more than 74 percentage points."),
    ("body",
     "The ordering of the four models is preserved on CICIoV2024, which matters more "
     "than the absolute figures: whatever the capture, the supervised baselines lose "
     "an unseen class and the residual gate is what recovers it, to the extent that "
     "the attack is recoverable at all."),
    ("sub", "E.  ATTACK-WISE DETECTION PERFORMANCE"),
    ("body",
     "Aggregate numbers conceal the mechanism, so we report the residual by attack "
     "class and by arbitration identifier. The gate detects an attack if, and only if, "
     "that attack sends payload bytes the twin does not expect for that identifier. "
     "This is a real condition, not a formality, and Fig. 5 shows a case where it is "
     "satisfied and a case where it is not, on the same capture and the same model."),
    ("fig4", None),
    ("fcap",
     "FIGURE 5.  Mean digital-twin residual on the two GEM-CAN tampering identifiers. "
     "Brake tampering raises the residual by a factor of 9.5 and is detected. Steering "
     "tampering leaves it below the normal level and is not, because the forged "
     "payloads fall inside the range the twin already accepts for that identifier."),
    ("body",
     "Under brake tampering the mean residual on identifier 18FF00F9 is 25.13 bytes, "
     "9.5 times the 2.64 bytes measured on normal frames of that identifier, and the "
     "twin localises the attack precisely: bytes DATA_4 and DATA_5 carry almost all of "
     "the increase. On 18FF70F9 the mean residual under steering tampering is 3.06 "
     "bytes, below the 6.54 bytes of normal traffic on the same identifier. The forged "
     "steering payloads lie inside the range the twin already accepts, so the gate "
     "cannot detect steering tampering. That is not an experiment that failed; it is a "
     "measurement. A score above learnt-normal is a positive deviation, and a positive "
     "deviation is what the gate looks for."),
    ("body",
     "This restriction is worth stating exactly, because it is the one most easily "
     "overstated. Detection does not require knowing what the attack is, but it does "
     "require the attack to be anomalous in the payload. Those are different "
     "conditions, and conflating them would overstate the method."),
    ("sub", "F.  FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS"),
    ("body",
     "The cost of the second channel is its false-positive rate. An always-on alarm "
     "detects every attack, so a detection rate without a false-positive rate is not a "
     "result. On normal GEM-CAN test traffic the proposed model raises false alarms on "
     "2.26% of frames, against 0.03% to 0.31% for the baselines: about 340 false "
     "alarms per 15,063 normal frames. The target rate at calibration is 1%, and the "
     "realised rate is higher for two reasons: false alarms from the supervised head "
     "are counted as alarms of the system, and the normal traffic used for validation "
     "is not the normal traffic used for test."),
    ("body",
     "Fig. 6 shows the whole trade-off rather than the single point. On CICIoV2024 "
     "the curve is effectively a step: detection reaches 100% at a false-alarm rate "
     "below the sweep resolution and stays there, with an area under the curve of "
     "1.000 for all three quarantined classes, so the operating point reported in "
     "Table VIII is conservative and could be lowered by an order of magnitude at no "
     "cost in detection. The GEM-CAN run available to us recorded the operating point "
     "only and not the full sweep, so the corresponding curve for that capture is "
     "left for future work rather than estimated here."),
    ("body",
     "The operationally meaningful unit is alarms per unit time. At 2.26%, a bus "
     "carrying f frames per second produces 0.0226f alarms per second. At any frame "
     "rate a production powertrain bus sustains, that is more alarms than an operator "
     "or a downstream controller could triage. We therefore present the gate at this "
     "operating point as a research instrument rather than a deployable alarm. The "
     "threshold is the knob available, and a deployment moves down the detection-cost "
     "curve by lowering the target false-positive rate; what the tolerable rate is "
     "depends on a per-vehicle frame rate and a per-fleet triage budget that these "
     "captures do not supply."),
    ("body",
     "The false negatives are concentrated and explicable rather than scattered. On "
     "GEM-CAN every missed frame of the quarantined class is zero, because the gate "
     "detects all 119; the systematic false negative is steering tampering, which the "
     "gate misses entirely for the reason given in Section VII-E. On CICIoV2024 there are no false negatives at all: every frame of every quarantined class raises an alarm, and the false-alarm rate falls to 0.37%."),
    ("fig6", None),
    ("fcap",
     "FIGURE 6.  Detection of the quarantined class against false alarms on normal "
     "traffic, as the gate threshold is swept, on CICIoV2024. The three curves "
     "coincide and are drawn with different dash patterns for that reason; the area "
     "under each is 1.000. The circle marks the operating point reported in Table "
     "VIII, which is conservative: detection is already complete at a far smaller "
     "false-alarm rate."),
    ("sub", "G.  ABLATION STUDY"),
    ("body",
     "Table V isolates the contribution of each component on the same CNN-BiLSTM "
     "backbone. The ladder is not monotonic with the identifier included, and we read "
     "it as it stands: macro-F1 falls from 0.9979 to 0.9958 when the scalar twin "
     "output is added, and only the full per-byte residual recovers and surpasses the "
     "backbone at 0.9989. With the identifier excluded the ladder does rise at each "
     "step, from 0.9914 to 0.9965 to 0.9979. In both variants adaptive fusion adds "
     "nothing beyond the residual, 0.9979 to 0.9979."),
    ("body",
     "The conclusion we draw is narrower than the architecture would invite. In this "
     "experiment the gain lies in the residual, not in the attention layer, and we "
     "report it that way. The decomposition of Section VII-A reinforces it: on the "
     "quarantined class the attention-weighted fusion contributes nothing at all, "
     "because the detection comes entirely from the gate."),
    ("sub", "H.  EXPLAINABILITY ANALYSIS"),
    ("body",
     "Every alarm raised by the gate carries three pieces of evidence without any "
     "additional model: which of the two views fired, which arbitration identifier "
     "carried the largest standardised residual, and which payload byte within it. On "
     "GEM-CAN brake tampering this localisation is unambiguous, with bytes DATA_4 and "
     "DATA_5 of identifier 18FF00F9 carrying almost the whole deviation, which "
     "corresponds to the forged brake command. An operator receiving such an alarm is "
     "told the ECU and the signal, not merely that something is wrong."),
    ("body",
     "Shapley additive explanation values [29] computed over the supervised branch "
     "attribute its decisions across payload bytes, identifier and timing features, "
     "and confirm that with the identifier available the model leans heavily on it for "
     "the volumetric class, which is the quantitative form of the shortcut documented "
     "in Section IV. With the identifier removed the attributions redistribute onto "
     "the payload bytes. Both tools identify where the evidence lies rather than why a "
     "payload is surprising; the latter would require the vehicle's signal database, "
     "as noted in Section V-F."),
    ("sub", "I.  COMPUTATIONAL EFFICIENCY"),
    ("body",
     "A forward pass through all four models plus one gate evaluation takes 12.0 ms "
     "per frame on CPU, approximately 3 ms per model. The gate itself is two "
     "standardisations and a comparison and is negligible beside the twin's forward "
     "pass. The digital twin adds one GRU step per frame over the supervised branch, "
     "which is the price of the open-set capability."),
    ("body",
     "We do not claim line-rate operation on vehicle hardware. The system was not "
     "designed for an embedded target, the measurement is single-threaded CPU on a "
     "development machine, and a production deployment would need quantisation, "
     "batching across identifiers, or an accelerator. What the figure does establish "
     "is that the residual channel is not the bottleneck: the cost is dominated by the "
     "supervised backbones that any CAN detector of this family already pays."),
]

SEC8 = [
    ("sec", "VIII.  DISCUSSION"),
    ("sub", "A.  KEY FINDINGS"),
    ("body",
     "Three findings follow from the experiments. First, closed-set accuracy has "
     "stopped discriminating between CAN intrusion detectors on these benchmarks: four "
     "architectures agree to within 0.0031 macro-F1, and a lookup table on the "
     "arbitration identifier alone reaches 98.9% accuracy on GEM-CAN. Reporting "
     "accuracy as the headline on such a benchmark measures the dataset, not the "
     "model."),
    ("body",
     "Second, the same four models separate by more than 74 percentage points when one "
     "attack class is quarantined from every training stage, and the separation is "
     "attributable to one component. The supervised head contributes none of the "
     "detection of the unseen class, because it has no label to assign; the "
     "digital-twin residual gate contributes all of it. A channel fitted without "
     "labels generalises to an attack nobody labelled, and a channel fitted with "
     "labels does not."),
    ("body",
     "Third, that generalisation has a precise and measurable boundary. The gate "
     "detects an attack if and only if the attack drives the payload outside the range "
     "the twin has learned for that identifier. Brake tampering on GEM-CAN raises the "
     "residual by a factor of 9.5 and is detected completely; steering tampering on "
     "the same capture lowers it and is missed completely. The condition is a property "
     "of the attack, not of the detector's capacity, and no additional model depth "
     "changes it."),
    ("sub", "B.  CROSS-DATASET GENERALIZATION"),
    ("body",
     "The method transfers across both real vehicles in this study. On GEM-CAN it detects the quarantined brake-tampering class completely where the best baseline reaches 25.21%, and on CICIoV2024 it detects every one of three quarantined classes completely at a false-alarm rate of 0.37%. Two captures, two manufacturers, two attack families, one protocol, and the same outcome."),
    ("body",
     "The useful conclusion from the cross-dataset work is therefore not a performance "
     "number but a predicate. Before deploying a residual gate on a new vehicle or a "
     "new capture, one can ask whether the attacks of concern place implausible values "
     "on the bus. If they do, the gate transfers. If they hold signals at plausible "
     "values, it does not, and no amount of retraining on normal traffic will change "
     "that, because the twin is already modelling normality correctly. This is a "
     "sharper and more portable statement than an accuracy figure on a second dataset."),
    ("sub", "C.  PRACTICAL IMPLICATIONS FOR EV CYBERSECURITY"),
    ("body",
     "For an electric vehicle fleet, three implications follow. A residual gate of "
     "this kind is best understood as a second line rather than a replacement: it "
     "should run alongside a supervised classifier, which remains the precise and "
     "cheap detector for attacks that have been catalogued, and should be the "
     "component that catches what the catalogue misses. Deploying the gate alone would "
     "trade precision for coverage unnecessarily."),
    ("body",
     "The false-alarm budget is the binding constraint, not accuracy. At a 2.26% rate "
     "on a bus carrying thousands of frames per second, the alarm stream exceeds any "
     "plausible triage capacity, so an operational deployment must either lower the "
     "target rate and accept reduced sensitivity, or aggregate alarms over a window "
     "before escalating. The per-identifier, per-byte localisation described in "
     "Section VII-H is what makes such aggregation tractable, because alarms can be "
     "grouped by the ECU and signal they implicate rather than counted in bulk."),
    ("body",
     "Finally, the attacks that matter most on an electric, drive-by-wire platform are "
     "precisely the hard ones. A denial-of-service flood is disruptive and easy to "
     "detect; a forged steering or torque command is dangerous and, when its values "
     "are plausible, invisible to a payload residual. A deployment that must cover "
     "that case needs a timing or sequence model alongside the twin, because the "
     "evidence for such an attack is in when and how often a frame arrives rather than "
     "in what it contains."),
    ("sub", "D.  LIMITATIONS AND THREATS TO VALIDITY"),
    ("body",
     "The method has two boundaries. Detection requires payload anomaly, as "
     "established in Section VII-E, and a replay attack would evade the gate for the "
     "same reason: replaying a genuine payload presents the twin with a predictable "
     "payload, so the residual stays low. Replay detection is timing or sequence "
     "modelling rather than payload modelling. A third case, an attack arriving on an "
     "arbitration identifier the twin has never observed, was untested in earlier work and is resolved here: every CICIoV2024 attack identifier is absent from the benign "
     "capture, and the gate detects all three such classes completely, because the twin "
     "holds no normal profile for them and the standardised residual is correspondingly "
     "large."),
    ("body",
     "The evaluation has four limits of its own. The block-wise split leaves adjacent "
     "blocks temporally correlated, and 87.4% of GEM-CAN frames are byte-identical to "
     "another frame, with 80.6% of test frames having an exact twin in training; both "
     "inflate frame-level metrics equally for all models, so the leave-one-attack-out "
     "comparison is unaffected, but both should temper any reading of the absolute "
     "closed-set figures. All figures come from a single seeded run: the 74-point gap "
     "that carries the argument is far larger than any plausible seed variance, while "
     "the differences of a few thousandths in Tables IV and V are not, and we claim no "
     "ordering among the four architectures. The CICIoV2024 quarantine was repeated "
     "over four seeds, which bounds that concern there but not on GEM-CAN, where a "
     "repeat would require access to the raw capture. The baseline set contains three "
     "supervised architectures and does not include the unsupervised comparisons a "
     "reader might reasonably expect: a plain autoencoder, a one-class model, an "
     "explicit open-set scoring rule, or a per-identifier range check, which is the "
     "simplest thing that could detect brake tampering on this capture. And validation "
     "covers two real captures plus one synthetic benchmark; Car-Hacking, OTIDS and "
     "ROAD remain untested and would each require the same conversion and the same "
     "four-stage quarantine before they could be reported."),
    ("body",
     "The most serious threat to validity is the redundancy of the benchmarks "
     "themselves, quantified in Section III-C. When 99.9% of a capture’s frames are "
     "byte-identical to another frame and the whole release holds 3,588 distinct "
     "identifier-payload tuples, a frame-level score close to 1.000 is substantially "
     "a measurement of the benchmark. This affects every detector evaluated on these "
     "captures, ours included, and it is the reason we lead with the "
     "leave-one-attack-out alarm rate rather than with closed-set accuracy, and the "
     "reason we treat GEM-CAN rather than CICIoV2024 as the load-bearing "
     "evaluation. A capture with per-frame labels, recorded timestamps and "
     "substantially more distinct payloads would be needed to resolve detection "
     "capability beyond the level these benchmarks can support."),
    ("body",
     "A further threat is specific to CICIoV2024 and is restated here so that it "
     "cannot be overlooked: the release carries no timestamp, so the inter-arrival "
     "feature is synthesised and constant on that dataset. Any conclusion that depends "
     "on timing is therefore not supported by the CICIoV2024 results, and the "
     "comparison in Section VII-C is read as a comparison of payload-based detection "
     "only."),
]

SEC9 = [
    ("sec", "IX.  CONCLUSION AND FUTURE WORK"),
    ("body",
     "Closed-set evaluation has told the CAN intrusion detection literature that the "
     "problem is nearly solved. The four architectures compared here sit within 0.0031 "
     "macro-F1 of one another on the dataset studied, which says more about the "
     "benchmark than about the models, and a lookup table on the arbitration "
     "identifier alone reaches 98.9% accuracy on the same capture."),
    ("body",
     "The same four models separate by more than 74 percentage points when one attack "
     "class is quarantined from the label space, from the digital twin's history and "
     "from the anomaly threshold's calibration set. The digital-twin residual gate, "
     "calibrated on normal traffic alone, raises an alarm on every held-out frame of "
     "the quarantined class at a 2.26% false-positive rate, where the best supervised "
     "baseline reaches 25.21%. The mechanism has a well-defined boundary and we have "
     "measured it rather than asserted it: detection rests on the attack being "
     "anomalous in the payload, and an attack whose forged values fall within the "
     "learnt-normal range is not detected. Evaluation on a second real capture, "
     "CICIoV2024 from a 2019 Ford, and on the synthetic SynCAN benchmark, tests that "
     "boundary rather than merely repeating the headline figure."),
    ("body",
     "Future work follows the limitations directly. The evaluation should be extended "
     "across vehicles and drive cycles, and across the remaining public captures, so "
     "that the transfer predicate of Section VIII-B can be tested rather than "
     "inferred. The baseline set should be widened to include a per-identifier range "
     "check, an autoencoder and an explicit open-set scoring rule, so that the "
     "residual gate is measured against the simplest alternatives as well as against "
     "supervised ones. The false-alarm rate needs to come down by an order of "
     "magnitude before the gate is deployable, which points at alarm aggregation over "
     "a window and at a better residual representation. And the attacks the residual "
     "cannot see, those that mimic normal payload statistics, require a timing or "
     "sequence channel alongside the twin; that is where the remaining headroom lies."),
]
