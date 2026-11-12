# -*- coding: utf-8 -*-
"""
Matched-false-alarm comparison for a leave-one-attack-out run.

The headline comparison in the paper reads each detector at its own operating
point: the supervised baselines emit a hard argmax and alarm on roughly 0.1% of
normal frames, while the proposed model's gate is calibrated to a far looser
one. A reviewer is right to ask what the baselines detect when they are allowed
the same false-alarm budget. This script answers that.

Each baseline is scored continuously as s = 1 - P(Normal). Its threshold is set
to the quantile of s over normal test frames that yields the target false-alarm
rate, and detection of the quarantined class is then read at that threshold.

It reads the artefacts the last `train_zeroday.py` run left in saved_models/,
so run it straight after that run and against the same GEMCAN_DIR.

    GEMCAN_DIR=data/CICIoV2024_as_GEM python matched_fpr.py \
        --unknown "Steering Tampering" --target-fpr 0.0037
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from models.anomaly_gate import ResidualAnomalyGate
from preprocessing.loader import CLASS_NAMES, CLASS_TO_IDX
from preprocessing.prepare import prepare
from preprocessing.windowing import build_ecu_streams
from train_zeroday import ZD_MODELS, build_model, predict, zd_path

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unknown", default="Steering Tampering", choices=CLASS_NAMES[1:])
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="false-alarm budget every model is held to. Default: the "
                         "rate the proposed model's own gate realised in that run.")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "matched_fpr.json"))
    args = ap.parse_args()

    unknown = CLASS_TO_IDX[args.unknown]
    known = [c for c in CLASS_NAMES if c != args.unknown]
    remap = {CLASS_TO_IDX[c]: i for i, c in enumerate(known)}

    p = prepare()
    y, gather = p.windows.y, p.windows.gather_idx
    te = np.where(p.windows.split == "test")[0]
    y_small = np.full(len(y), -1, dtype=np.int64)
    for o, n in remap.items():
        y_small[y == o] = n

    residual = np.load(zd_path("residuals.npy"))
    id_index = np.ascontiguousarray(
        p.df["arbitration_id"].map(lambda v: p.id_vocab.get(int(v), 0)).to_numpy(),
        dtype=np.int64)
    z = np.load(zd_path("ecu_streams.npz"))
    streams, mask = z["streams"], z["mask"]
    gate = ResidualAnomalyGate.load(zd_path("gate.npz"))

    base = p.features(True)
    X_raw, X_fusion = base, np.concatenate([base, residual], axis=1).astype(np.float32)

    y_test = y[te]
    is_norm, is_unk = y_test == 0, y_test == unknown
    print(f"held out: {args.unknown}   test frames: {len(te):,}  "
          f"normal {int(is_norm.sum()):,}  held-out {int(is_unk.sum()):,}")

    # the proposed model's own realised false-alarm rate, unless one is given
    gate_alarm = gate.alarm(residual[te], id_index[te], streams[te], mask[te])
    target = args.target_fpr
    rows = []
    for name, key, mode, uses_gate in ZD_MODELS:
        w = zd_path(f"{key}.pt")
        if not os.path.exists(w):
            raise SystemExit(f"missing weights: {w}\nRun train_zeroday.py first; this "
                             f"script never estimates a number it cannot compute.")
        X = X_fusion if mode == "fusion" else X_raw
        fusion = key == "twinshield"
        model = build_model(key, X.shape[1], len(known), streams.shape[1])
        model.load_state_dict(torch.load(w, map_location="cpu"))
        proba = predict(model, X, gather, y_small, te,
                        streams=streams if fusion else None,
                        mask=mask if fusion else None)
        s = 1.0 - proba[:, 0]                       # index 0 is Normal in both spaces

        own = ((proba.argmax(1) != 0) | gate_alarm) if uses_gate else (proba.argmax(1) != 0)
        own_fpr = float(own[is_norm].mean())
        own_det = float(own[is_unk].mean())
        if target is None and uses_gate:
            target = own_fpr
        rows.append([name, s, own_fpr, own_det, uses_gate])

    if target is None:
        raise SystemExit("no target false-alarm rate: pass --target-fpr")
    print(f"matched false-alarm budget: {target*100:.2f}%\n")
    print(f"{'Model':<24}{'own FPR':>9}{'own det':>9}{'matched thr':>13}"
          f"{'matched FPR':>13}{'matched det':>13}")
    out = {"unknown": args.unknown, "target_fpr": target, "models": {}}
    for name, s, own_fpr, own_det, uses_gate in rows:
        thr = float(np.quantile(s[is_norm], 1.0 - target))
        alarm = s > thr
        m_fpr, m_det = float(alarm[is_norm].mean()), float(alarm[is_unk].mean())
        print(f"{name:<24}{own_fpr*100:>8.2f}%{own_det*100:>8.2f}%{thr:>13.6f}"
              f"{m_fpr*100:>12.2f}%{m_det*100:>12.2f}%")
        out["models"][name] = {"own_false_positive_rate": own_fpr,
                               "own_detection_rate": own_det,
                               "matched_threshold": thr,
                               "matched_false_positive_rate": m_fpr,
                               "matched_detection_rate": m_det}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
