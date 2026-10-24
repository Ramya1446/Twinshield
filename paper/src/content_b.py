# -*- coding: utf-8 -*-
"""Sections 3 (Related Work) and 4 (Datasets and Problem Formulation)."""

SEC3 = [
    ("sec", "III.  RELATED WORK"),
    ("body",
     "Research on in-vehicle intrusion detection divides along two axes: what the "
     "detector learns from, and what it is evaluated against. We review the field "
     "along the first axis in Sections III-A to III-C and along the second in Section "
     "III-D, because the second is where this paper's contribution sits."),
    ("sub", "A.  CAN BUS INTRUSION DETECTION"),
    ("body",
     "The earliest detectors used properties of the bus rather than of the payload. "
     "Muter and Asaj [14] applied entropy to the message stream, on the reasoning that "
     "injected traffic lowers the entropy of an otherwise highly regular bus. Lee et "
     "al. [13] built OTIDS around the remote-frame response offset, exploiting the "
     "fixed timing with which an ECU answers a request. Cho and Shin [18] went below "
     "the protocol entirely and fingerprinted ECUs from clock skew, so that a frame "
     "claiming an identifier its usual transmitter did not send can be recognised. "
     "These methods need no labelled attacks and are cheap, but each covers one "
     "narrow mechanism: entropy and timing detect volumetric injection well and a "
     "low-rate payload forgery on a legitimate identifier hardly at all."),
    ("body",
     "Classical supervised learning followed, on handcrafted frame features. Bari et "
     "al. [6] compared support vector machines, decision trees and k-nearest "
     "neighbours on real vehicle captures, and Lokman et al. [8] surveyed the family. "
     "Accuracies are high and the models are interpretable, but the features are "
     "chosen for the attacks in hand and the detectors inherit the closed-set limit "
     "discussed in Section II-C."),
    ("sub", "B.  LEARNING-BASED CAN TRAFFIC ANOMALY DETECTION"),
    ("body",
     "Deep models removed the feature-engineering step. Kang and Kang [5] applied a "
     "deep neural network to CAN payloads; Song et al. [9] used a deep convolutional "
     "network over sequences of frames; Seo et al. [7] trained a generative "
     "adversarial network, GIDS, to recognise injected traffic. Rai et al. [10] "
     "compared LSTM, GRU and BiLSTM detectors on CAN captures and reported strong "
     "sequence-learning performance, and Wang et al. [11] added data augmentation for "
     "connected autonomous vehicles. Lightweight and adaptive variants continue the "
     "line: Xu et al. [17] proposed a dynamically adaptive detection framework "
     "intended for the compute budget an in-vehicle unit can actually carry."),
    ("body",
     "A second strand represents normal traffic and thresholds the deviation from it. "
     "Taylor et al. [20] trained a recurrent network to predict the next payload of an "
     "identifier and flagged large prediction errors. Shahriar et al. [19] built "
     "CANShield around multi-view reconstruction of the signal space. The closest "
     "antecedent to this work is CANet [21], which learns a joint model of CAN signals "
     "with per-identifier subnetworks and scores reconstruction error. These methods "
     "are label-free and therefore not bounded by an attack catalogue, which is the "
     "property this paper builds on."),
    ("sub", "C.  DIGITAL TWIN-BASED INTRUSION DETECTION"),
    ("body",
     "A digital twin is a model kept in correspondence with a physical system [22]. In "
     "industrial control, twins have been used to detect intrusion by comparing the "
     "plant's observed behaviour against the behaviour the twin predicts, so that the "
     "detector is defined by a model of correct operation rather than by a library of "
     "faults. The automotive literature has largely used twins for prognostics, "
     "calibration and fleet analytics rather than for security, and where behavioural "
     "deviation is used for CAN security it is usually a reconstruction error rather "
     "than a twin maintained per ECU."),
    ("body",
     "The distinction matters for this paper. A twin fitted on attack-free traffic "
     "defines normality per arbitration identifier, so its residual is defined for any "
     "frame whatsoever, including a frame belonging to an attack class that has never "
     "been labelled. That is the property a closed-set classifier cannot have, and it "
     "is why the residual rather than the classifier carries the zero-day result in "
     "Section VII."),
    ("sub", "D.  CROSS-DATASET EVALUATION OF CAN INTRUSION DETECTION SYSTEMS"),
    ("body",
     "Public captures for CAN intrusion detection include Car-Hacking and OTIDS from "
     "HCRL [13], the ROAD dataset of Verma et al. [23], the synthetic SynCAN benchmark "
     "released with CANet [21], GEM-CAN from an autonomous electric vehicle [24], and "
     "CICIoV2024 from the Canadian Institute for Cybersecurity [25]. They differ in "
     "vehicle, bus speed, identifier set, attack family and recorded fields, and that "
     "heterogeneity is the obstacle to cross-dataset work: a model trained on one "
     "capture's identifier vocabulary cannot be applied directly to another's."),
    ("body",
     "Most published detectors are validated on a single capture. Where a second "
     "dataset is used, it is normally used to repeat a performance number rather than "
     "to test a hypothesis about when the method works. Table I summarises the studies "
     "discussed above by method, by the dataset each was evaluated on, and by whether "
     "any attack class was held out of training. The pattern is uniform: high "
     "accuracy, one or two datasets, and no held-out class."),
    ("tlabel", "TABLE I"),
    ("tcap", "Summary of representative CAN intrusion detection studies"),
    ("keeptable", 0),
    ("body",
     "Three observations follow, and together they form the gap this paper addresses. "
     "The accuracies are recognition rates inside a fixed catalogue. The unsupervised "
     "entries are trained on attack-free traffic but reported separately from the "
     "supervised ones, so the comparison that a deployment needs is never made on one "
     "split. And validation rarely crosses a dataset boundary, so the conditions under "
     "which a method transfers are unknown. We place a supervised classifier and an "
     "attack-agnostic residual gate on the same data, under one quarantine applied at "
     "four stages, on two real vehicle captures, and report both under a metric that "
     "is defined irrespective of label space."),
]

SEC4 = [
    ("sec", "IV.  DATASETS AND PROBLEM FORMULATION"),
    ("sub", "A.  GEM-CAN DATASET"),
    ("body",
     "GEM-CAN [24] is a public capture recorded by Tavasoli et al. from a Polaris GEM "
     "e6, an autonomous electric vehicle, on a closed rural test track. The vehicle "
     "carries a 250 kbit/s CAN bus and the traffic was recorded through a PCAN-USB "
     "interface connected via a PACMod drive-by-wire interface. Two captures are "
     "supplied: one of nominal autonomous driving, and one during which attacks were "
     "physically injected onto the live bus while the vehicle was stationary in "
     "autonomous mode, for safety. Every frame carries a timestamp, an arbitration "
     "identifier, a data length code, eight payload bytes and an attack-type label."),
    ("body",
     "The capture suits the question of this paper for three reasons. It is not a "
     "simulation, so the payload dynamics are real. It is an electric, drive-by-wire "
     "platform, so the commands that are attacked are the commands that actuate the "
     "vehicle. And it contains both a volumetric attack and two low-rate tampering "
     "attacks on legitimate identifiers, which lets one dataset exercise both the easy "
     "and the difficult detection regimes. Its composition is given in Table II."),
    ("tlabel", "TABLE II"),
    ("tcap", "Composition of the GEM-CAN capture"),
    ("keeptable", 1),
    ("body",
     "The two attack families differ in a way that determines everything that follows. "
     "The denial-of-service attack transmits arbitration identifier 00000000. Because "
     "CAN arbitration is won by the lowest identifier, that frame takes priority over "
     "all legitimate traffic and the bus load rises from 20% to 100% during the flood. "
     "It is trivially separable: identifier 00000000 is sent by no legitimate ECU on "
     "this vehicle, so the identifier alone identifies the attack. The tampering "
     "attacks insert a small number of forged frames on 18FF00F9 and 18FF70F9, "
     "identifiers that carry real traffic throughout the capture, and the recorded "
     "effect is a steering-lock delay of about 20 seconds. Here the identifier is "
     "uninformative as a discriminator and detection must rest on the payload."),
    ("sub", "B.  CICIoV2024 DATASET"),
    ("body",
     "CICIoV2024 [25] is a capture released by the Canadian Institute for "
     "Cybersecurity from a 2019 Ford. It contains benign traffic and five attack "
     "classes: a denial-of-service flood and spoofing of the gas, RPM, speed and "
     "steering-wheel signals. The published composition is 1,223,737 benign frames, "
     "74,663 DoS frames, 9,991 gas-spoofing, 54,900 RPM-spoofing, 24,951 "
     "speed-spoofing and 19,977 steering-wheel-spoofing frames. The spoofing classes "
     "are the interesting ones for this paper, because each forges a signal on an "
     "identifier that already carries legitimate traffic, which is the regime in which "
     "the arbitration identifier carries no information."),
    ("body",
     "Two properties of the release shape how it can be used, and both are stated here "
     "rather than left implicit. First, the schema is identifier, eight payload bytes "
     "and three label columns; it carries neither a timestamp nor a data length code. "
     "We therefore synthesise a timestamp from row order at a nominal one-millisecond "
     "interval and fix the data length code at eight. The consequence is that the "
     "inter-arrival feature is constant on this dataset and carries no information, so "
     "no timing-based claim is made on it. Second, the label space of this work has "
     "three attack slots and CICIoV2024 has five attack classes, so the four spoofing "
     "classes are grouped: steering-wheel spoofing occupies one slot on its own, and "
     "gas, RPM and speed spoofing are grouped as powertrain spoofing in the other. The "
     "assignment is recorded in machine-readable form by the converter and every "
     "result in Section VII-B is reported under the CICIoV2024 names."),
    ("sub", "C.  DATASET CHARACTERISTICS AND ATTACK CATEGORIES"),
    ("body",
     "Table III sets the two captures side by side. They agree on the structural "
     "features that matter to the method and disagree on almost everything else, which "
     "is what makes the pair useful: GEM-CAN is an electric, drive-by-wire research "
     "platform with four classes and a severe identifier shortcut, while CICIoV2024 is "
     "a production passenger vehicle with six classes, no timestamp and a far larger "
     "benign population."),
    ("body",
     "Before training anything we audited both captures for shortcuts, and found a "
     "severe one in GEM-CAN. A lookup table that guesses the majority class of each "
     "arbitration identifier, using no other information and no learning whatsoever, "
     "reaches 98.9% accuracy on the test split. It scores a per-class F1 of 1.00 on "
     "DoS and 0.00 on both tampering classes, giving a macro-F1 of 0.498. Two things "
     "follow, and they shape the remainder of this paper. This is not a dataset with "
     "a meaningful accuracy headline, so macro-F1 and per-class figures are reported "
     "throughout. And the usefulness of any payload-modelling component has to be "
     "judged on the tampering classes, because DoS is already solved by a lookup "
     "table. We therefore retrain every model with the arbitration identifier removed "
     "from the input as well, which bounds how far the shortcut can be credited to a "
     "model rather than recognised as a property of the capture."),
    ("body",
     "A second property of GEM-CAN is relevant to every figure that follows: the data "
     "are highly redundant. 87.4% of frames are byte-identical to at least one other "
     "frame, and 80.6% of test frames are exactly identical to a frame in the training "
     "split. This is intrinsic to CAN, where ECUs emit fixed status payloads at fixed "
     "intervals, but it makes any frame-level metric look better than the underlying "
     "difficulty warrants, and is another reason not to lead with accuracy."),
    ("tlabel", "TABLE III"),
    ("tcap", "Characteristics of the two evaluation datasets"),
    ("table", "DATASET_CHARS"),
    ("body",
     "Across both captures the attacks fall into two categories, and the distinction "
     "is more useful than the dataset-specific names. A volumetric attack injects "
     "frames at high rate, usually on an identifier that is either illegitimate or "
     "rarely used, and changes the statistics of the bus as a whole. A tampering or "
     "spoofing attack writes forged payloads onto an identifier that carries real "
     "traffic, at a rate low enough to leave the bus statistics intact, and changes "
     "only what the receiving ECU is told. Volumetric attacks are easy for any "
     "detector, including a lookup table. Spoofing attacks are the ones that require a "
     "model of what the payload should have been, and they are the attacks against "
     "which this paper measures itself."),
    ("sub", "D.  PROBLEM DEFINITION AND THREAT MODEL"),
    ("body",
     "Let a capture be a sequence of frames f_t = (tau_t, i_t, d_t, p_t), where tau_t "
     "is the timestamp, i_t the arbitration identifier, d_t the data length code and "
     "p_t the eight payload bytes. Each frame carries a label y_t from a finite set "
     "that contains Normal and a number of attack classes. The detector observes a "
     "context window of the W frames up to and including f_t and must emit a decision "
     "for f_t."),
    ("body",
     "We ask two questions of that detector. The closed-set question is the usual one: "
     "given every class at training time, predict y_t. The open-set question is the "
     "one this paper is built around: one attack class is designated unknown and "
     "removed from every fitting stage, and the detector, which has no output index "
     "for it, must still answer whether frame f_t should raise an alarm. Because one "
     "model then has a label the others lack, accuracy and F1 are not comparable "
     "across models in that setting; the alarm rate is, because it is defined in any "
     "label space."),
    ("body",
     "The adversary is assumed to have already obtained write access to the bus, "
     "whether through a physical diagnostic port, a compromised telematics unit or an "
     "aftermarket device, which is the access level demonstrated in [2], [3] and [4]. "
     "The adversary can jam the bus, and can transmit arbitrary payloads on "
     "identifiers that legitimate ECUs already use. The detector is passive: it "
     "observes the bus and cannot authenticate senders, because the protocol provides "
     "no mechanism for it. Attacks that do not modify the bus, such as passive "
     "eavesdropping, are outside this threat model, as is an attack arriving on an "
     "identifier the twin has never observed, which is noted as a limitation in "
     "Section VIII-D."),
]
