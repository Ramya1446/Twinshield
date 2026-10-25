# -*- coding: utf-8 -*-
"""Sections 1-4 of the restructured paper."""

ABSTRACT = (
    "An electric vehicle is a cyber-physical system: the commands that brake it, "
    "steer it and drive it are software messages on a Controller Area Network (CAN) "
    "bus that carries no sender identity and no authentication. An adversary who "
    "reaches that bus, through the diagnostic port, a compromised telematics unit or "
    "an aftermarket dongle, can forge any of those commands, and the vehicle obeys "
    "them. The protocol cannot be repaired without replacing every electronic control "
    "unit on the vehicle, so intrusion detection is the only defence deployable on "
    "vehicles already on the road. Published CAN detectors report accuracies above "
    "99%, but they are trained and tested against a fixed catalogue of attacks: an "
    "attack outside that catalogue has no output class to fall into, so the detector "
    "reports it as normal traffic. This paper presents TwinShield-Fusion, which pairs "
    "a supervised CNN-BiLSTM classifier with a gated recurrent unit digital twin "
    "fitted on attack-free traffic alone. The twin predicts the next payload of each "
    "arbitration identifier, and the standardised per-identifier, per-byte prediction "
    "residual is thresholded at a calibrated false-positive rate, so the detector "
    "measures deviation from normal vehicle behaviour rather than similarity to a "
    "known attack. We evaluate on two real captures: GEM-CAN, 143,444 frames from a "
    "Polaris GEM e6 electric vehicle, and CICIoV2024, a capture from a 2019 Ford. "
    "With one attack class quarantined from every training stage on GEM-CAN, the "
    "three supervised baselines detect 0.84%, 0.00% and 25.21% of that class while "
    "TwinShield-Fusion detects 100.00% of it at a 2.26% false-positive rate. On "
    "CICIoV2024 the same protocol, applied once to each of three attack classes, "
    "detects 100.00% of each at a 0.37% false-positive rate. Every one of these "
    "figures is specific to the class named as quarantined in that run; none of them "
    "is a claim of universal zero-day detection. The boundary is measured rather than "
    "assumed, and we report it: on the same GEM-CAN capture the gate detects none of "
    "the steering-tampering class, whose forged payloads lie inside the range the "
    "twin already accepts, and on SynCAN, a synthetic benchmark composed entirely of "
    "attacks of that kind, it detects at most 7.07%. We further show that both public "
    "captures are far more redundant than their size suggests, CICIoV2024 containing "
    "only 3,588 distinct identifier-payload tuples across 1.4 million frames, so "
    "frame-level scores on them measure the benchmark as much as the detector."
)

INDEX = ("Automotive cyber-physical security, controller area network, digital twin, "
         "electric vehicles, intrusion detection systems, residual anomaly detection, "
         "zero-day attacks.")

SEC2 = [
    ("sec", "II.  INTRODUCTION"),
    ("sub", "A.  BACKGROUND AND MOTIVATION"),
    ("body",
     "An electric vehicle is not a mechanical machine with software bolted on. Its "
     "brake pedal is a sensor, its steering column is a motor, and its accelerator is "
     "a request. Between the driver and the wheels sit dozens of electronic control "
     "units (ECUs) that exchange several thousand messages per second over a "
     "Controller Area Network (CAN) bus standardised as ISO 11898-1 [1]. Electrified "
     "and drive-by-wire platforms deepen that dependence: regenerative braking blends "
     "friction and motor torque in software, electric power steering has no mechanical "
     "fallback, and the battery management system governs the current that moves the "
     "vehicle at all. Every one of those functions is actuated by a CAN frame."),
    ("body",
     "CAN was designed in the 1980s for a closed network of trusted nodes. A frame "
     "carries an arbitration identifier, a data length code and up to eight payload "
     "bytes. It carries no sender address, no signature and no sequence number, and "
     "every node receives every frame. Arbitration is won by the numerically lowest "
     "identifier, so a node that transmits identifier zero continuously starves every "
     "other transmitter on the bus. None of this is a defect of an implementation; it "
     "is the protocol, and it is the same protocol on vehicles sold today."),
    ("body",
     "The consequences have been demonstrated repeatedly on production vehicles. "
     "Koscher et al. [2] showed that an attacker with bus access could disable the "
     "brakes and the engine of a modern car. Checkoway et al. [3] showed that the "
     "access itself can be obtained remotely, without physical contact. Miller and "
     "Valasek [4] drove an unaltered passenger vehicle off the road over a cellular "
     "link. Industry reporting has tracked the shift: an analysis of more than nine "
     "hundred automotive cybersecurity incidents found that 84.5% were carried out "
     "remotely [12]. An electric vehicle widens the attack surface further, because "
     "charging infrastructure, over-the-air update channels and telematics units are "
     "all network-facing and all eventually speak to the same internal bus."),
    ("body",
     "The structural problem is that the bus cannot be fixed. Adding authentication to "
     "CAN means adding cryptographic material and processing budget to every ECU on "
     "the vehicle, which is neither retrofittable to the vehicles already on the road "
     "nor free on the ones being built. Intrusion detection is therefore not one "
     "option among several; for the installed fleet it is the only option. The "
     "question this paper asks is what such a detector must be able to do before it "
     "can be trusted on a vehicle."),
    ("sub", "B.  RESEARCH CHALLENGES"),
    ("body",
     "Five properties of the problem make CAN intrusion detection harder than its "
     "reported accuracies suggest."),
    ("body",
     "The attack catalogue is open-ended. A detector is trained on the attacks its "
     "authors could record. The adversary is under no obligation to use them. A "
     "supervised classifier has one output neuron per class it was taught, so an "
     "attack it has never seen cannot be assigned to a class that does not exist; in "
     "practice it is assigned to the majority class, which is normal traffic. The "
     "failure is silent, and it is the failure that matters operationally."),
    ("body",
     "Benchmarks contain shortcuts. Volumetric attacks dominate the public captures, "
     "and a volumetric attack usually announces itself in the arbitration identifier. "
     "Section IV-C shows that on GEM-CAN a lookup table over the identifier alone "
     "reaches 98.9% accuracy, so a deep model reported at 99% may have learned almost "
     "nothing a lookup table did not already know."),
    ("body",
     "Normal traffic is repetitive and the classes are imbalanced. ECUs emit fixed "
     "status payloads at fixed intervals, so consecutive frames are nearly identical "
     "and a uniformly random split leaks near-duplicates between training and test. "
     "On GEM-CAN the ratio of normal to tampering frames is approximately 152:1, so "
     "an uninformative detector that always answers normal scores well on accuracy."),
    ("body",
     "False alarms have a budget. A vehicle bus carries thousands of frames per "
     "second, so a false-positive rate that looks small in a table becomes an alarm "
     "stream no operator or downstream controller can triage. Any detector that buys "
     "sensitivity must report what that sensitivity costs."),
    ("body",
     "Datasets are heterogeneous. Captures differ in vehicle, bus speed, identifier "
     "set, attack family and even in the fields they record; CICIoV2024, used in this "
     "paper, ships no timestamp at all. A method that is only shown to work on the "
     "capture it was designed around has not been shown to generalise."),
    ("sub", "C.  RESEARCH GAP"),
    ("body",
     "Three gaps follow from those challenges and are visible in the literature "
     "summarised in Section III. First, evaluation is closed-set. The accuracies "
     "reported are recognition rates for attacks inside the catalogue, and they say "
     "nothing about attacks outside it, yet the headline claim is usually stated "
     "without that qualification. Second, supervised and unsupervised detectors are "
     "reported in separate papers, on separate splits, under separate metrics, so the "
     "comparison that a deployment actually needs, between a detector that names "
     "attacks and one that measures deviation, is never made on the same data. Third, "
     "validation is usually confined to one capture, and the few cross-dataset studies "
     "that exist report performance rather than the conditions under which a method "
     "transfers."),
    ("sub", "D.  RESEARCH CONTRIBUTIONS"),
    ("body",
     "This paper makes the following contributions."),
    ("body",
     "1) TwinShield-Fusion, a detector that combines a supervised CNN-BiLSTM "
     "classifier with a GRU digital twin of normal CAN behaviour. The standardised "
     "per-identifier, per-byte prediction residual is read through two views, a "
     "per-frame view and a per-ECU-stream view, and thresholded at a false-positive "
     "rate calibrated on normal traffic alone. The residual gate is the component we "
     "claim; the attention-weighted multi-ECU fusion is retained from the backbone "
     "and we show in Section VI-G that it contributes no measurable gain in these "
     "experiments."),
    ("body",
     "2) A leave-one-attack-out protocol with a four-stage quarantine. The held-out "
     "class is removed from the label space, from every training context window, from "
     "the digital twin's per-ECU history and from the gate's calibration set, so no "
     "component sees a single frame of it during fitting."),
    ("body",
     "3) An arbitration-identifier bias audit that quantifies the shortcut in the "
     "benchmark itself, together with an identifier-free control variant of every "
     "experiment."),
    ("body",
     "4) Evaluation on two real vehicle captures, GEM-CAN from a Polaris GEM e6 "
     "electric vehicle and CICIoV2024 from a 2019 Ford, under the same protocol, with "
     "a converter that maps a third-party capture onto the schema this pipeline reads."),
    ("body",
     "5) An explicit, measured statement of the method's operating condition: the "
     "residual gate detects attacks that place implausible values on the bus, and does "
     "not detect attacks whose forged values lie inside the learnt-normal range. We "
     "report the case where our own method fails rather than omitting it."),
    ("body",
     "6) The false-alarm cost of the second channel, expressed as a rate, as alarms "
     "per unit time on a bus of a given frame rate, and as a detection-against-false-alarm curve over the whole threshold range, so that the operating point can be "
     "judged against a triage budget rather than against a single table entry."),
    ("body",
     "7) A redundancy and leakage audit of both public captures, which finds that "
     "87.4% of GEM-CAN frames and 99.9% of CICIoV2024 frames are byte-identical to "
     "another frame, and that the whole CICIoV2024 release contains 3,588 distinct "
     "identifier-payload tuples in 1.4 million frames. Frame-level scores on these "
     "benchmarks have to be read in that light, including ours."),
]
