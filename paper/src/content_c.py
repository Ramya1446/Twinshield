# -*- coding: utf-8 -*-
"""Sections 5 (Proposed Framework) and 6 (Experimental Setup)."""

SEC5 = [
    ("sec", "V.  PROPOSED TWINSHIELD-FUSION FRAMEWORK"),
    ("sub", "A.  SYSTEM ARCHITECTURE"),
    ("fig1", None),
    ("body",
     "Fig. 1 shows the architecture. Raw CAN traffic is preprocessed once and read by "
     "two branches that answer different questions. The supervised branch builds a "
     "32-frame context window and classifies the current frame with a CNN-BiLSTM "
     "enhanced by an attention-weighted fusion of per-ECU residual streams; it answers "
     "which known attack this is. The digital-twin branch predicts the next payload of "
     "each arbitration identifier from normal traffic alone and standardises the "
     "prediction residual per identifier and per byte; it answers how far this frame "
     "is from what the vehicle normally does. A gate thresholds the standardised "
     "residual at a false-positive rate calibrated on normal frames, and an alarm is "
     "raised if either branch fires."),
    ("body",
     "The separation is the point of the design. The supervised branch is bounded by "
     "its label set and cannot generalise beyond the classes it was fitted on. The "
     "twin branch is fitted without any label at all, so its residual is defined for "
     "every frame, including one belonging to an attack class nobody has named. The "
     "detector can therefore say that something is wrong even when it cannot say what "
     "is wrong, which is the behaviour an operational system needs and a closed-set "
     "classifier cannot provide."),
    ("sub", "B.  CAN TRAFFIC PREPROCESSING"),
    ("body",
     "Each frame is represented by its eight payload bytes scaled to [0,1], its data "
     "length code, the inter-arrival gap to the previous frame on the same identifier, "
     "and an index for the arbitration identifier itself. A single frame is not enough "
     "to distinguish a forged command from a legitimate one on the same identifier, so "
     "the supervised branch reads a context window of 32 frames ending at the frame "
     "under test, in bus order. The unit of classification is therefore the window, "
     "and the contents of the window also dictate what must be excluded under the "
     "quarantine protocol of Section VI-C."),
    ("body",
     "On CICIoV2024 the inter-arrival gap is constant, because the release carries no "
     "timestamp and one is synthesised from row order. The feature is retained so that "
     "the two datasets pass through identical code, but it contributes nothing there, "
     "and the comparison in Section VII-C is read with that in mind."),
    ("sub", "C.  DIGITAL TWIN-BASED TRAFFIC PREDICTION"),
    ("body",
     "Let p_t in [0,1]^8 be the scaled payload of frame t and i_t its arbitration "
     "identifier. The twin is a gated recurrent unit [26] with 96 hidden units and a "
     "16-dimensional embedding of the identifier, which lets one network serve every "
     "ECU on the bus while still conditioning on which ECU is speaking. It reads the "
     "eight most recent payloads observed on identifier i_t and predicts the next one. "
     "Training minimises mean squared error against the true next payload, over normal "
     "frames in the training split only."),
    ("body",
     "The quantity the rest of the method uses is the residual"),
    ("eq", "r_t = | p_t − p̂_t |     (1)"),
    ("body",
     "taken element-wise, so r_t is an eight-dimensional vector with one component per "
     "payload byte. On GEM-CAN the twin reaches a mean absolute error of 0.041 in the "
     "units it optimises, or 10.52 on the 0-255 byte scale, measured on held-out "
     "normal frames. That is not a small error, and it does not need to be: detection "
     "rests not on a residual of zero but on a residual that is reliably different "
     "between normal and tampered traffic on the same identifier. CAN payloads "
     "multiplex independent signals, and some bytes are rolling counters or noisy "
     "sensor readings that are not predictable in principle."),
    ("sub", "D.  RESIDUAL GENERATION AND ANOMALY DETECTION"),
    ("body",
     "The gate is the component that does not depend on known attacks. It reads the "
     "same residual through two views. The frame view asks how far this one frame "
     "departs from what its own identifier normally looks like, byte by byte:"),
    ("eq", "z_frame(t) = max_b [ r_{t,b} − μ_{i,b} ] / σ_{i,b}     (2)"),
    ("body",
     "where i is the identifier of the frame and b indexes the eight payload bytes. "
     "This is the sharp channel: a forged payload that moves one byte far outside its "
     "usual range shows up immediately and is not averaged away by surrounding "
     "traffic. The ECU view asks how far a whole identifier's stream has drifted "
     "across the context window:"),
    ("eq", "z_ecu(t) = max_{s present} [ v̄_s(t) − μ_s ] / σ_s     (3)"),
    ("body",
     "where v̄_s(t) is the mean residual of stream s inside the window. This is the "
     "sustained channel: it survives a single quiet frame and reflects the multi-ECU "
     "structure the fusion branch is built on. An alarm is raised when the larger of "
     "the two crosses a threshold:"),
    ("eq", "alarm(t) = 1 [ max(z_frame, z_ecu) > z* ]     (4)"),
    ("body",
     "Per-identifier, per-byte standardisation is what makes a single bus-wide "
     "threshold usable. Some ECUs and some bytes sit at a high baseline residual "
     "because they are intrinsically hard to predict; judging each byte of each "
     "identifier against its own normal behaviour prevents those from firing "
     "constantly. The means and standard deviations are estimated on normal frames of "
     "the training split, and z* is the (1 - alpha) quantile of the score over normal "
     "frames of the validation split, with alpha the target false-positive rate. No "
     "attack frame of any class takes part in fitting or calibrating the gate, and "
     "that is the whole basis for calling it attack-agnostic: there is no signature in "
     "it to memorise, only a model of normal."),
    ("sub", "E.  INTRUSION CLASSIFICATION AND DECISION MECHANISM"),
    ("body",
     "The supervised branch is a convolutional network followed by a bidirectional "
     "long short-term memory network [27]. Three one-dimensional convolutional layers "
     "with 64 filters, kernels of width five and then three, read the context window "
     "and extract local byte patterns that are invariant to their position in the "
     "window, which is what a burst of injected frames looks like. The BiLSTM, with 96 "
     "hidden units in each direction, then reads the convolutional feature sequence "
     "both ways, so a frame is judged against traffic that follows it as well as "
     "traffic that precedes it."),
    ("body",
     "Each arbitration identifier is treated as a separate ECU stream. Within a "
     "context window, stream s is summarised by its mean residual, giving a tensor "
     "over streams, and streams absent from the window are masked out. Additive "
     "attention [28] weights the streams,"),
    ("eq", "e_s = vᵀ tanh(W h_s + b),   α = softmax(e)     (5)"),
    ("body",
     "and the fused vector sum_s alpha_s h_s is concatenated with the CNN-BiLSTM "
     "embedding before a softmax head over the trained classes. This branch names an "
     "attack and is bounded by its label set in exactly the way described in Section "
     "II-B; we treat it as the recognition channel and make no claim that it "
     "generalises beyond the classes it was fitted on."),
    ("body",
     "The attention-weighted fusion is described here for completeness because it is "
     "part of the implementation, but it is not claimed as a source of the results. "
     "The ablation of Section VI-G measures its contribution directly and finds it to "
     "be nil in both input variants, 0.9979 to 0.9979 with the identifier and 0.9979 "
     "to 0.9979 without it, and the decomposition of Section VI-A shows that it "
     "contributes none of the quarantined-class detection either. The component this "
     "paper claims is the residual gate of Section IV-D; readers reproducing the "
     "method should expect the residual, not the attention layer, to carry the "
     "result."),
    ("body",
     "The two branches are joined by disjunction. A frame raises an alarm if the "
     "supervised head assigns it any class other than Normal, or if the gate fires. "
     "The disjunction is deliberate: the classifier is precise on what it knows and "
     "blind to the rest, the gate is broad and noisier, and a detector that must not "
     "miss an unknown attack should take the union rather than the intersection. The "
     "cost of that choice is quantified in Section VII-F."),
    ("sub", "F.  MODEL EXPLAINABILITY"),
    ("body",
     "An alarm that cannot be explained cannot be triaged. The framework exposes two "
     "complementary explanations. The gate reports, for every alarm, which arbitration "
     "identifier and which payload byte carried the largest standardised residual, and "
     "which of the two views fired; that localises an alarm to an ECU and a signal "
     "without any additional model. For the supervised branch we compute Shapley "
     "additive explanation values [29] over its inputs, which attribute a decision "
     "across payload bytes, identifier and timing features."),
    ("body",
     "Both explanations are partial in the same way, and it is worth stating. They "
     "identify where the evidence is, not why the twin finds a given payload "
     "surprising. A full account of the latter would require decoding the payload into "
     "physical signals, which needs the vehicle's signal database and is outside the "
     "scope of a detector that must work without one."),
]

SEC6 = [
    ("sec", "VI.  EXPERIMENTAL SETUP"),
    ("sub", "A.  EXPERIMENTAL ENVIRONMENT"),
    ("body",
     "All models are implemented in PyTorch and trained and evaluated on CPU. The "
     "main results use a fixed random seed of 42, and the leave-one-attack-out "
     "experiment on CICIoV2024 is additionally repeated with seeds 1, 2 and 3 so "
     "that the figures in Section VI-B carry an uncertainty rather than resting on "
     "one draw. No GPU is used at any stage, which is deliberate: the "
     "runtime figures in Section VII-I are then measured on hardware comparable to "
     "what a vehicle or a workshop diagnostic tool could plausibly carry, rather than "
     "on an accelerator that no production ECU contains."),
    ("sub", "B.  DATA PREPROCESSING AND FEATURE ALIGNMENT"),
    ("body",
     "Both captures are read into one schema: timestamp, arbitration identifier, data "
     "length code, eight payload bytes and a class label. GEM-CAN is used as "
     "published. CICIoV2024 is mapped onto the same schema by a converter that "
     "synthesises the timestamp from row order, fixes the data length code at eight, "
     "assigns each source class to a label slot as described in Section IV-B, and "
     "records the assignment in a machine-readable file so that no figure drawn from "
     "the converted data can silently inherit the other dataset's class names. The "
     "converter verifies its own output by loading it back through the project's "
     "loader and refusing to proceed if any slot is empty."),
    ("body",
     "From that common schema the feature matrix is identical for both datasets: eight "
     "payload bytes scaled to [0,1], the data length code, the inter-arrival gap and "
     "an identifier index, assembled into 32-frame context windows. The identifier "
     "vocabulary is built per dataset, since the two vehicles share no identifiers."),
    ("sub", "C.  TRAINING, VALIDATION, AND TESTING PROTOCOL"),
    ("body",
     "Consecutive CAN frames are almost identical, so a uniformly random split leaks "
     "near-duplicate frames between training and test and inflates every metric. "
     "Frames are therefore cut into contiguous blocks of 256 and whole blocks are "
     "assigned to training, validation and test in a 70/15/15 ratio, stratified so "
     "that every class is represented in each split. No frame's context window crosses "
     "a split boundary. This does not remove the correlation between adjacent blocks, "
     "and that residual limitation is reported in Section VIII-D rather than hidden."),
    ("body",
     "The leave-one-attack-out protocol is the central experimental instrument. One "
     "attack class is designated unknown and removed at four stages: the detector head "
     "is built with one fewer output so there is no index for it; every training and "
     "validation frame whose context window contains a frame of that class is dropped; "
     "the digital twin is refitted from scratch on normal frames whose per-ECU history "
     "is free of it; and the gate's means, standard deviations and threshold are taken "
     "from normal frames only. The test split is left untouched and still contains "
     "every frame of the held-out class."),
    ("sub", "D.  BASELINE MODELS"),
    ("body",
     "Three supervised baselines are trained on the same splits, with the same class "
     "weights and the same evaluation code: a one-dimensional CNN, a BiLSTM, and a "
     "CNN-BiLSTM, the last of which is also the backbone of the proposed model. Two "
     "intermediate variants are reported for the ablation of Section VII-G: the "
     "backbone with a scalar twin output appended, and the backbone with the full "
     "per-byte residual vector appended. Every model is additionally re-run with the "
     "arbitration identifier removed from the input, because the audit of Section IV "
     "shows the identifier alone is nearly sufficient on GEM-CAN."),
    ("body",
     "All detectors use a 32-frame context window, a batch size of 512, the Adam "
     "optimiser at a learning rate of 10^-3, and class-weighted cross-entropy to "
     "compensate for the imbalance between normal and tampering traffic. The digital "
     "twin is trained for five epochs and the detectors for six, and the weights kept "
     "are those with the best validation macro-F1, which are then evaluated once on "
     "the test split."),
    ("sub", "E.  EVALUATION METRICS"),
    ("body",
     "For the closed-set setting we report macro-F1, which weights every class equally "
     "and therefore does not let the dominant normal class conceal failure on the "
     "tampering classes, together with per-class F1 where the detail matters."),
    ("body",
     "For the open-set setting macro-F1 across models is meaningless, because the "
     "proposed model has a label the baselines lack. We therefore report the alarm "
     "rate: the fraction of frames of a given true class on which the detector raised "
     "an alarm of any kind. On the held-out attack class this is the detection rate, "
     "and on the normal class it is the false-positive rate. Both are defined "
     "identically for every model regardless of label space, which is the property "
     "that makes the comparison fair. We additionally report the known-class macro-F1 "
     "of each model on the classes it did retain, so that a model cannot buy detection "
     "by degrading into an always-alarm detector without that being visible."),
]
