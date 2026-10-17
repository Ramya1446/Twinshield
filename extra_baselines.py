# -*- coding: utf-8 -*-
"""
Unsupervised and open-set baselines under the same leave-one-attack-out
quarantine as train_zeroday.py.

The paper compares three supervised architectures against TwinShield-Fusion.
That is not the comparison a reader asks for, because the proposed system's
second channel is unsupervised and none of the baselines have one. This script
adds the missing column:

    per-ID range check    the simplest thing that could work: flag a payload
                          byte never seen on that identifier in training
    isolation forest      a standard one-class detector on frame features
    autoencoder           reconstruction error on frame features, normal only
    twin residual alone   the proposed gate with the supervised head removed
    MSP / energy          open-set scores read off each supervised baseline

Every detector that has a threshold is calibrated the same way the gate is:
on held-out NORMAL validation frames at a common target false-positive rate,
with no attack frame of any class taking part. The quarantined class is removed
from training exactly as train_zeroday.py removes it, through the same
clean_masks() call, so the comparison is like for like.

    python extra_baselines.py                       # reuses the saved zero-day run
    python extra_baselines.py --target-fpr 0.01
    python extra_baselines.py --skip autoencoder    # if torch is unavailable

Requires train_zeroday.py to have been run first: it reuses that run's twin,
residuals, ECU streams, gate and detectors rather than refitting them, so the
numbers sit beside the ones already in results/zero_day.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(ROOT, "saved_models")
RESULTS = os.path.join(ROOT, "results")

from models.anomaly_gate import ResidualAnomalyGate            # noqa: E402
from preprocessing.loader import CLASS_NAMES, CLASS_TO_IDX, DATA_COLS  # noqa: E402
from preprocessing.prepare import prepare                      # noqa: E402
from preprocessing.windowing import build_id_history           # noqa: E402
from train_zeroday import ZD_MODELS, clean_masks, zd_path      # noqa: E402


# ---------------------------------------------------------------------------
# Calibration: one rule for every detector that has a threshold.
# ---------------------------------------------------------------------------
def calibrate(score_cal: np.ndarray, target_fpr: float) -> float:
    """The (1 - target) quantile of the score over normal validation frames."""
    return float(np.quantile(score_cal, 1.0 - target_fpr))


def rates(alarm: np.ndarray, y_test: np.ndarray, unknown: int) -> Dict[str, float]:
    normal = y_test == 0
    unk = y_test == unknown
    out = {
        "unknown_detection_rate": float(alarm[unk].mean()) if unk.any() else 0.0,
        "false_positive_rate": float(alarm[normal].mean()) if normal.any() else 0.0,
    }
    for c, name in enumerate(CLASS_NAMES):
        sel = y_test == c
        if sel.any():
            out[f"alarm_rate::{name}"] = float(alarm[sel].mean())
    return out


# ---------------------------------------------------------------------------
# Baseline 1: per-identifier byte range check
# ---------------------------------------------------------------------------
def range_check(payload: np.ndarray, id_index: np.ndarray, fit_idx: np.ndarray,
                n_ids: int) -> np.ndarray:
    """Flag a frame when any payload byte falls outside the interval that byte
    occupied on that identifier in normal training traffic.

    No threshold and nothing to calibrate, which is the point: it is the
    cheapest detector that could plausibly catch a payload-tampering attack,
    and the proposed method has to beat it to be worth its cost.
    """
    dim = payload.shape[1]
    lo = np.full((n_ids, dim), np.inf, dtype=np.float64)
    hi = np.full((n_ids, dim), -np.inf, dtype=np.float64)
    np.minimum.at(lo, id_index[fit_idx], payload[fit_idx])
    np.maximum.at(hi, id_index[fit_idx], payload[fit_idx])
    # An identifier never seen as normal in training has an empty interval; a
    # frame on it is outside by definition, which is the honest reading.
    out_lo = payload < lo[id_index]
    out_hi = payload > hi[id_index]
    return (out_lo | out_hi).any(axis=1)


# ---------------------------------------------------------------------------
# Baseline 2: isolation forest on frame features
# ---------------------------------------------------------------------------
def isolation_forest(X: np.ndarray, fit_idx: np.ndarray, cal_idx: np.ndarray,
                     te_idx: np.ndarray, target_fpr: float, seed: int):
    from sklearn.ensemble import IsolationForest
    n_fit = min(len(fit_idx), 60_000)          # the estimator saturates well
    rng = np.random.default_rng(seed)          # before the full normal split
    sub = rng.choice(fit_idx, n_fit, replace=False) if len(fit_idx) > n_fit else fit_idx
    clf = IsolationForest(n_estimators=200, max_samples=min(4096, len(sub)),
                          random_state=seed, n_jobs=-1).fit(X[sub])
    s_cal = -clf.score_samples(X[cal_idx])
    s_te = -clf.score_samples(X[te_idx])
    thr = calibrate(s_cal, target_fpr)
    return s_te > thr, thr


# ---------------------------------------------------------------------------
# Baseline 3: autoencoder on frame features
# ---------------------------------------------------------------------------
def autoencoder(X: np.ndarray, fit_idx: np.ndarray, cal_idx: np.ndarray,
                te_idx: np.ndarray, target_fpr: float, seed: int,
                epochs: int = 8, batch: int = 512):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = X.shape[1]
    mu = X[fit_idx].mean(0, keepdims=True)
    sd = X[fit_idx].std(0, keepdims=True) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)

    net = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
        nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    order = np.array(fit_idx)
    for ep in range(epochs):
        np.random.default_rng(seed + ep).shuffle(order)
        tot = 0.0
        for s in range(0, len(order), batch):
            xb = torch.from_numpy(Z[order[s:s + batch]]).to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), xb)
            loss.backward(); opt.step()
            tot += float(loss) * len(xb)
        print(f"      autoencoder epoch {ep + 1}/{epochs}  mse {tot / len(order):.5f}")

    @torch.no_grad()
    def err(idx):
        out = []
        for s in range(0, len(idx), 4096):
            xb = torch.from_numpy(Z[idx[s:s + 4096]]).to(dev)
            out.append(((net(xb) - xb) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out)

    thr = calibrate(err(cal_idx), target_fpr)
    return err(te_idx) > thr, thr


# ---------------------------------------------------------------------------
# Baseline 5: open-set scores off the supervised baselines
# ---------------------------------------------------------------------------
def open_set_scores(logits: np.ndarray) -> Dict[str, np.ndarray]:
    """Two standard rules. Both are high when the model is unsure, so both are
    thresholded from above like every other score here."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    msp = 1.0 - p.max(axis=1)                      # maximum softmax probability
    energy = -(logits.max(axis=1) +                # free energy, numerically safe
               np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1)))
    return {"MSP": msp, "Energy": energy}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unknown", default=None,
                    help="class to quarantine; defaults to the one in zero_day.json")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-id", dest="use_id", action="store_true", default=True)
    ap.add_argument("--no-use-id", dest="use_id", action="store_false")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="baselines to leave out, e.g. autoencoder openset")
    ap.add_argument("--out", default=os.path.join(RESULTS, "extra_baselines.json"))
    args = ap.parse_args()

    zd_file = os.path.join(RESULTS, "zero_day.json")
    if not os.path.exists(zd_file):
        raise SystemExit("results/zero_day.json not found. Run:  python train_zeroday.py")
    zd = json.load(open(zd_file, encoding="utf-8"))
    unknown_name = args.unknown or zd["unknown_class"]
    unknown = CLASS_TO_IDX[unknown_name]

    print("=" * 74)
    print(f"Unsupervised and open-set baselines -- '{unknown_name}' quarantined")
    print("=" * 74)

    print("\n[1/4] Preparing dataset and reapplying the quarantine ...", flush=True)
    p = prepare()
    y = p.windows.y
    split = p.windows.split
    id_index = np.ascontiguousarray(
        p.df["arbitration_id"].map(lambda v: p.id_vocab.get(int(v), 0)).to_numpy(),
        dtype=np.int64)
    n_ids = len(p.id_vocab) + 1
    id_hist = build_id_history(p.df, p.windows.block_id)
    window_clean, history_clean = clean_masks(p, id_hist, unknown)

    is_tr, is_va, is_te = split == "train", split == "val", split == "test"
    fit_idx = np.where(is_tr & (y == 0) & window_clean & history_clean)[0]
    cal_idx = np.where(is_va & (y == 0) & window_clean)[0]
    te_idx = np.where(is_te)[0]
    y_test = y[te_idx]
    print(f"      fit {len(fit_idx):,} normal / calibrate {len(cal_idx):,} normal "
          f"/ test {len(te_idx):,}  ({int((y_test == unknown).sum()):,} held-out "
          f"{unknown_name})")

    payload = p.df[DATA_COLS].to_numpy(dtype=np.float32) / 255.0
    X = p.features(args.use_id)
    out: Dict[str, Dict] = {}

    print("\n[2/4] Unsupervised baselines ...", flush=True)
    t0 = time.time()
    alarm = range_check(payload, id_index, fit_idx, n_ids)[te_idx]
    out["Per-ID range check"] = {**rates(alarm, y_test, unknown),
                                 "threshold": None, "seconds": round(time.time() - t0, 1)}

    if "isolation" not in args.skip:
        t0 = time.time()
        alarm, thr = isolation_forest(X, fit_idx, cal_idx, te_idx,
                                      args.target_fpr, args.seed)
        out["Isolation forest"] = {**rates(alarm, y_test, unknown),
                                   "threshold": thr, "seconds": round(time.time() - t0, 1)}

    if "autoencoder" not in args.skip:
        t0 = time.time()
        try:
            alarm, thr = autoencoder(X, fit_idx, cal_idx, te_idx,
                                     args.target_fpr, args.seed)
            out["Autoencoder"] = {**rates(alarm, y_test, unknown), "threshold": thr,
                                  "seconds": round(time.time() - t0, 1)}
        except ImportError:
            print("      [skip] autoencoder needs torch")

    # the proposed gate, standing on its own with no supervised head beside it
    gate_path = zd_path("gate.npz")
    if os.path.exists(gate_path):
        gate = ResidualAnomalyGate.load(gate_path)
        residual = np.load(zd_path("residuals.npy"))
        st = np.load(zd_path("ecu_streams.npz"))
        alarm = gate.alarm(residual[te_idx], id_index[te_idx],
                           st["streams"][te_idx], st["mask"][te_idx])
        out["Twin residual alone"] = {**rates(alarm, y_test, unknown),
                                      "threshold": float(gate.threshold), "seconds": 0.0}
    else:
        print(f"      [skip] {gate_path} not found")

    print("\n[3/4] Open-set scores on the supervised baselines ...", flush=True)
    if "openset" not in args.skip:
        try:
            scored = supervised_open_set(p, X, id_index, y, unknown, fit_idx,
                                         cal_idx, te_idx, args, window_clean)
            out.update(scored)
        except ImportError:
            print("      [skip] open-set scores need torch")

    print("\n[4/4] Writing results ...", flush=True)
    payload_out = {
        "unknown_class": unknown_name, "target_fpr": args.target_fpr,
        "seed": args.seed, "use_id": args.use_id,
        "test_support": {n: int((y_test == c).sum()) for c, n in enumerate(CLASS_NAMES)
                         if (y_test == c).any()},
        "baselines": out,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload_out, fh, indent=2)

    print(f"\n  Detection of the quarantined '{unknown_name}' against false alarms")
    print("    {:<26}{:>12}{:>16}".format("Detector", "Detected", "False alarms"))
    for name, r in out.items():
        print("    {:<26}{:>12}{:>16}".format(
            name, f"{r['unknown_detection_rate'] * 100:.2f}%",
            f"{r['false_positive_rate'] * 100:.2f}%"))
    for name, r in zd.get("models", {}).items():
        print("    {:<26}{:>12}{:>16}   (from zero_day.json)".format(
            name, f"{r['unknown_detection_rate'] * 100:.2f}%",
            f"{r['false_positive_rate'] * 100:.2f}%"))
    print("\n  wrote", args.out)


def supervised_open_set(p, X, id_index, y, unknown, fit_idx, cal_idx, te_idx,
                        args, window_clean) -> Dict[str, Dict]:
    """Read MSP and energy off each supervised detector the zero-day run saved."""
    import torch
    from train_zeroday import build_model, predict

    y_test = y[te_idx]
    # `unknown` is a class index, so it has to be resolved to a name before it
    # can be compared with CLASS_NAMES; comparing the two directly kept every
    # class and built a k-way head against a (k-1)-way checkpoint.
    unknown_name = CLASS_NAMES[unknown]
    known_names = [c for c in CLASS_NAMES if c != unknown_name]
    remap = {CLASS_TO_IDX[c]: i for i, c in enumerate(known_names)}
    y_small = np.full(len(y), -1, dtype=np.int64)
    for orig, new in remap.items():
        y_small[y == orig] = new

    residual = np.load(zd_path("residuals.npy"))
    st = np.load(zd_path("ecu_streams.npz"))
    streams, mask = st["streams"], st["mask"]
    X_fusion = np.concatenate([X, residual], axis=1).astype(np.float32)
    gather = p.windows.gather_idx

    res: Dict[str, Dict] = {}
    for name, key, mode, _ in ZD_MODELS:
        ckpt = zd_path(f"{key}.pt")
        if not os.path.exists(ckpt):
            print(f"      [skip] {ckpt} not found")
            continue
        Xi = X_fusion if mode == "fusion" else X
        fusion = key == "twinshield"
        model = build_model(key, Xi.shape[1], len(known_names), streams.shape[1])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        # predict() returns probabilities; logits are recovered as their log,
        # which leaves MSP exact and energy correct up to the constant that the
        # softmax dropped -- harmless, because the threshold is calibrated on
        # the same scale it is applied to.
        pr_cal = predict(model, Xi, gather, y_small, cal_idx,
                         streams=streams if fusion else None,
                         mask=mask if fusion else None)
        pr_te = predict(model, Xi, gather, y_small, te_idx,
                        streams=streams if fusion else None,
                        mask=mask if fusion else None)
        lg_cal = np.log(np.clip(pr_cal, 1e-12, None))
        lg_te = np.log(np.clip(pr_te, 1e-12, None))
        for rule, s_cal in open_set_scores(lg_cal).items():
            s_te = open_set_scores(lg_te)[rule]
            thr = calibrate(s_cal, args.target_fpr)
            res[f"{name} + {rule}"] = {**rates(s_te > thr, y_test, unknown),
                                       "threshold": float(thr), "seconds": 0.0}
        print(f"      {name}: MSP and energy scored")
    return res


if __name__ == "__main__":
    main()
