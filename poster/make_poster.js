const pptxgen = require("pptxgenjs");
const path = require("path");
const A = (f) => path.join(__dirname, "assets", f);

const NAVY = "4472C4", BLUE = "0070C0", INK = "231F20", GREY = "595959", WHITE = "FFFFFF";
const HEAD = "Arial", BODY = "Times New Roman";

const W = 14.0, H = 10.5, M = 0.12;
const HDR_H = 1.50;
const BODY_T = M + HDR_H + 0.10;
const BODY_B = H - M;
const COL_W = (W - 2 * M - 2 * 0.145) / 3;
const COLX = [M, M + COL_W + 0.145, M + 2 * (COL_W + 0.145)];
const PAD = 0.17;
const TW = COL_W - 2 * PAD;

const pres = new pptxgen();
pres.defineLayout({ name: "POSTER", width: W, height: H });
pres.layout = "POSTER";
pres.author = "TwinShield-Fusion";
pres.title = "TwinShield-Fusion project review poster";
const s = pres.addSlide();
s.background = { color: WHITE };

/* ------------------------------------------------------------------ header */
s.addShape(pres.ShapeType.rect, {
  x: M, y: M, w: W - 2 * M, h: HDR_H,
  fill: { color: WHITE }, line: { color: BLUE, width: 1 }, objectName: "headerFrame",
});
const LOGO_W = 3.42;
s.addImage({ path: A("vit_logo.png"), x: M + 0.22, y: M + (HDR_H - 2.98 * 247 / 939) / 2,
             w: 2.98, h: 2.98 * 247 / 939 });

const NX = M + LOGO_W;
s.addShape(pres.ShapeType.rect, {
  x: NX, y: M, w: W - M - NX, h: HDR_H,
  fill: { color: NAVY }, line: { color: NAVY, width: 1 }, objectName: "headerBand",
});
s.addText(
  "TWINSHIELD-FUSION: DIGITAL-TWIN RESIDUAL GATING FOR\nDETECTION OF UNSEEN ATTACKS ON AUTOMOTIVE CAN BUSES",
  { x: NX + 0.22, y: M + 0.09, w: 8.00, h: 0.62, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 15.5, bold: true, color: WHITE, lineSpacingMultiple: 0.98 });

const meta = [
  [{ text: "Student Names: ", options: { bold: true } },
   { text: "[Name 1 (Reg. No.)], [Name 2 (Reg. No.)], [Name 3 (Reg. No.)]" }],
  [{ text: "Guide Name: ", options: { bold: true } }, { text: "[Dr. Guide Name (Emp. ID)]" }],
  [{ text: "School: ", options: { bold: true } }, { text: "[School Name], VIT Chennai" }],
];
meta.forEach((runs, i) => {
  s.addText(runs, {
    x: NX + 0.22, y: M + 0.78 + i * 0.225, w: 8.00, h: 0.22, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 10.5, color: WHITE, valign: "middle",
  });
});
s.addImage({ path: A("vnest.png"), x: W - M - 2.26, y: M + (HDR_H - 1.98 * 176 / 396) / 2,
             w: 1.98, h: 1.98 * 176 / 396 });

/* ------------------------------------------------- column frames + stacking */
[0, 1, 2].forEach((i) =>
  s.addShape(pres.ShapeType.rect, {
    x: COLX[i], y: BODY_T, w: COL_W, h: BODY_B - BODY_T,
    fill: { color: WHITE }, line: { color: BLUE, width: 1 },
    objectName: `column${i + 1}`,
  }));

const cur = [BODY_T + PAD, BODY_T + PAD, BODY_T + PAD];

function heading(c, text) {
  s.addText(text, {
    x: COLX[c] + PAD, y: cur[c], w: TW, h: 0.26, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 13.5, bold: true, color: BLUE, valign: "middle",
  });
  cur[c] += 0.30;
}

function bullets(c, items, opts = {}) {
  const fs = opts.fontSize || 10;
  const lh = fs * 0.0158;
  const cpl = Math.floor((TW - 0.16) / (fs * 0.00595));
  let lines = 0;
  const runs = [];
  items.forEach((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    const plain = parts.map((r) => r.text).join("");
    lines += Math.max(1, Math.ceil(plain.length / cpl));
    parts.forEach((r, k) => {
      const o = Object.assign({}, r.options);
      o.bullet = { code: "2022", indent: 11 };
      o.breakLine = k === parts.length - 1 && i !== items.length - 1;
      runs.push({ text: r.text, options: o });
    });
  });
  const h = lines * lh + items.length * 0.05 + 0.05;
  s.addText(runs, {
    x: COLX[c] + PAD, y: cur[c], w: TW, h, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: fs, color: INK, align: "justify",
    lineSpacingMultiple: 0.95, paraSpaceAfter: 3,
  });
  cur[c] += h + 0.11;
}

function para(c, runs, fs = 10, gap = 0.11) {
  const plain = runs.map((r) => r.text).join("");
  const cpl = Math.floor(TW / (fs * 0.00595));
  const h = Math.max(1, Math.ceil(plain.length / cpl)) * fs * 0.0158 + 0.06;
  s.addText(runs, {
    x: COLX[c] + PAD, y: cur[c], w: TW, h, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: fs, color: INK, align: "justify", lineSpacingMultiple: 0.94,
  });
  cur[c] += h + gap;
}

function picture(c, file, wIn, ratio, caption) {
  const x = COLX[c] + PAD + (TW - wIn) / 2;
  const h = wIn * ratio;
  s.addImage({ path: A(file), x, y: cur[c], w: wIn, h });
  cur[c] += h + 0.04;
  if (caption) {
    s.addText(caption, {
      x: COLX[c] + PAD, y: cur[c], w: TW, h: 0.30, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9, italic: true, color: GREY, align: "center",
      lineSpacingMultiple: 0.92,
    });
    cur[c] += 0.30;
  }
  cur[c] += 0.10;
}

/* =============================== COLUMN 1 ================================= */
heading(0, "MOTIVATION / INTRODUCTION");
bullets(0, [
  "An electric vehicle carries its braking, steering and propulsion commands on the CAN bus, a broadcast network with no sender identity and no authentication.",
  "Any node on the bus can forge any message, and the protocol cannot change without replacing every ECU, so intrusion detection is the only practical defence.",
  "Published detectors are trained and tested on a fixed attack catalogue and report over 99% accuracy on it.",
  "An attack outside that catalogue has no output class to land on, so the model reports it as Normal. This is the zero-day failure.",
  "The benchmark is also weak: on GEM-CAN a lookup table on the arbitration identifier alone reaches 98.9% accuracy.",
]);

heading(0, "OBJECTIVES");
bullets(0, [
  "Fit a GRU digital twin that predicts the next CAN payload of each identifier from attack-free traffic only.",
  "Turn the prediction residual into a calibrated two-view anomaly gate, per frame and per ECU stream.",
  "Fuse that gate with a supervised CNN-BiLSTM classifier that names known attacks.",
  "Evaluate under a leave-one-attack-out protocol with a four-stage quarantine.",
  "Quantify the false-alarm cost and state where the method fails.",
  "Test whether the result survives a second, independent dataset.",
]);

heading(0, "SCOPE AND APPLICATIONS");
bullets(0, [
  "In-vehicle intrusion detection for electric and drive-by-wire platforms.",
  "Applies to any CAN capture with per-frame labels; a converter maps third-party datasets onto the same schema.",
  "Supports security-operations triage and post-incident forensics on vehicle fleets.",
  "A Streamlit application replays captures live, shows the residual per ECU and explains decisions with SHAP.",
]);

heading(0, "SDG ALIGNMENT");
para(0, [
  { text: "SDG 9 ", options: { bold: true } },
  { text: "- Industry, Innovation and Infrastructure: resilient safety-critical transport systems. " },
  { text: "SDG 11 ", options: { bold: true } },
  { text: "- Sustainable Cities and Communities: trustworthy electric public transport." },
]);

/* =============================== COLUMN 2 ================================= */
heading(1, "METHODOLOGY");
bullets(1, [
  [{ text: "Data: ", options: { bold: true } },
   { text: "GEM-CAN, 143,444 real frames from a Polaris GEM e6 electric vehicle - Normal, DoS, brake tampering and steering tampering." }],
  [{ text: "Features: ", options: { bold: true } },
   { text: "eight payload bytes scaled to [0,1], data length code, inter-arrival gap and identifier index, over a 32-frame window." }],
  [{ text: "Leakage control: ", options: { bold: true } },
   { text: "frames are cut into contiguous 256-frame blocks; whole blocks go to train, validation or test." }],
  [{ text: "Digital twin: ", options: { bold: true } },
   { text: "a GRU with 96 hidden units and a 16-dimensional identifier embedding, trained on normal frames only. The residual is r = | actual - predicted |, one value per payload byte." }],
  [{ text: "Anomaly gate: ", options: { bold: true } },
   { text: "a per-identifier, per-byte z-score and a per-ECU windowed z-score; an alarm is raised when the larger crosses a threshold." }],
  [{ text: "Calibration: ", options: { bold: true } },
   { text: "the threshold is the (1 - target FPR) quantile over held-out normal validation frames; no attack frame takes part." }],
  [{ text: "Supervised branch: ", options: { bold: true } },
   { text: "CNN, then BiLSTM, then attention-weighted multi-ECU fusion and a softmax head." }],
]);

heading(1, "SYSTEM ARCHITECTURE");
picture(1, "arch.png", TW, 887 / 1774,
  "Raw CAN traffic feeds two branches: the twin's residual is standardised and thresholded, the classifier names known attacks, and the gate combines both scores.");

heading(1, "ZERO-DAY EVALUATION PROTOCOL");
bullets(1, [
  "One attack class is removed from four places: the label space, every training context window, the twin's per-ECU history and the gate's calibration set.",
  "The test split is untouched and still holds all 119 brake-tampering frames.",
  "Since one model has a label the others lack, all four are judged on the question any detector must answer: did you raise an alarm?",
]);

/* =============================== COLUMN 3 ================================= */
heading(2, "RESULTS AND EVALUATION");
s.addTable([
  [{ text: "Model", options: { bold: true } },
   { text: "Closed-set\nmacro-F1", options: { bold: true } },
   { text: "Unseen attack\ndetected", options: { bold: true } },
   { text: "False\nalarms", options: { bold: true } }],
  ["CNN", "0.9968", "0.84%", "0.03%"],
  ["BiLSTM", "0.9948", "0.00%", "0.08%"],
  ["CNN-BiLSTM", "0.9979", "25.21%", "0.31%"],
  [{ text: "TwinShield-Fusion", options: { bold: true } },
   { text: "0.9979", options: { bold: true } },
   { text: "100.00%", options: { bold: true } },
   { text: "2.26%", options: { bold: true } }],
], {
  x: COLX[2] + PAD, y: cur[2], w: TW,
  colW: [1.42, 0.95, 1.02, 0.76],
  fontFace: BODY, fontSize: 9, color: INK, align: "center", valign: "middle",
  border: { type: "solid", color: "BFBFBF", pt: 0.5 },
  fill: { color: WHITE }, margin: 2, rowH: 0.20,
});
cur[2] += 0.20 * 5 + 0.12;
picture(2, "p_image3.png", 2.32, 1263 / 1819,
  "Leave-one-attack-out: brake tampering removed from every training stage; detection on its 119 held-out frames.");
bullets(2, [
  "On the closed set the four architectures span only 0.0031 macro-F1; under quarantine they separate by 74 points, and every detection comes from the gate, not the classifier.",
  "The residual rises 2.64 to 25.13 bytes on the brake identifier (9.5x) but falls on the steering identifier, which the gate misses. On SynCAN it stays near 1.04x and the gate detects at most 7.07%.",
], { fontSize: 9.5 });

heading(2, "CONCLUSION");
bullets(2, [
  "Closed-set accuracy no longer separates CAN intrusion detectors; a leave-one-attack-out protocol does.",
  "A twin fitted on normal traffic alone detects an attack class no component ever saw, at 2.26% false alarms and 12.0 ms per frame.",
  "Its boundary is measured, not assumed: implausible payloads are caught, plausible ones are not.",
], { fontSize: 9.5 });

heading(2, "REFERENCES");
bullets(2, [
  "ISO 11898-1, Road Vehicles - Controller Area Network, Part 1.",
  "K. Koscher et al., \"Experimental security analysis of a modern automobile,\" IEEE S&P, 2010.",
  "H. M. Song, J. Woo, H. K. Kim, \"In-vehicle network IDS using deep CNN,\" Veh. Commun., 2020.",
  "M. Hanselmann et al., \"CANet: Unsupervised IDS for CAN bus data,\" IEEE Access, 8:58194, 2020.",
  "M. Tavasoli et al., \"GEM-CAN: Real-world CAN-bus attack scenarios,\" 2024.",
], { fontSize: 8.5 });

heading(2, "CONTACT");
para(2, [
  { text: "Email: ", options: { bold: true } },
  { text: "[student@vitstudent.ac.in]   " },
  { text: "Mobile: ", options: { bold: true } },
  { text: "[+91 XXXXXXXXXX]" },
], 9.5, 0.0);

const LIMIT = BODY_B - PAD;
cur.forEach((y, i) =>
  console.log(`column ${i + 1}: fills to ${y.toFixed(2)}in of ${LIMIT.toFixed(2)}in` +
              (y > LIMIT ? `  <-- OVERFLOW by ${(y - LIMIT).toFixed(2)}in` : "")));

pres.writeFile({ fileName: path.join(__dirname, "TwinShield_Poster.pptx") })
  .then(() => console.log("written"))
  .catch((e) => { console.error(e); process.exit(1); });
