"""
Zero-day (unknown-attack) experiment -- the controlled version of the live demo.

The question
------------
Every baseline in this project is a supervised classifier over raw CAN traffic.
Supervised classifiers are bounded by their label set: an attack class that was
absent from training has no output neuron to land on, so the model must call it
something it has seen -- almost always "Normal".  TwinShield-Fusion carries an
extra channel, the Digital Twin residual, which is fitted on normal traffic only
and therefore measures deviation from normal rather than similarity to a known
attack.  Does that channel actually catch an attack nobody trained on?

The experiment
--------------
One attack class is designated *unknown* and removed from training completely:

  * the GRU Digital Twin is refitted from scratch, using only normal frames
    whose per-ECU history contains no frame of the unknown class;
  * the CNN, BiLSTM, CNN-BiLSTM and TwinShield-Fusion detectors are trained on a
    reduced label space that does not contain the unknown class at all, and on
    frames whose entire 32-frame context window is free of it;
  * the anomaly gate is fitted on normal training frames and its threshold set
    on normal validation frames at a target false-positive rate.

No component sees a single frame of the unknown class during fitting.  All four
models are then evaluated on the *untouched* test split, which still contains it.

Because one model has a label the others lack, the comparison is made on the
question every IDS actually has to answer -- **did you raise an alarm?** -- not
on which name was printed.

    python train_zeroday.py                              # Brake Tampering unknown
    python train_zeroday.py --unknown "Steering Tampering"
    python train_zeroday.py --epochs 3                   # quick rehearsal run

Run `python train.py` first: this script reuses its prepared-dataset cache.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate, per_class_f1
from models.anomaly_gate import ResidualAnomalyGate, sweep
from models.bilstm import BiLSTMDetector
from models.cnn import CNNDetector
from models.cnn_bilstm import CNNBiLSTMDetector
from models.digital_twin import compute_residuals, predict_payload
from models.residual_fusion import TwinShieldFusion
from preprocessing.loader import CLASS_NAMES, CLASS_TO_IDX, DATA_COLS
from preprocessing.prepare import Prepared, prepare
from preprocessing.windowing import build_ecu_streams, build_id_history
from train import SeqDataset, _run_epoch, set_seed, train_digital_twin

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(ROOT, "saved_models")
RESULTS = os.path.join(ROOT, "results")
ZD_FILE = os.path.join(RESULTS, "zero_day.json")
DEVICE = "cpu"

# (display name, architecture key, feature mode, carries the anomaly gate)
ZD_MODELS: List[Tuple[str, str, str, bool]] = [
    ("CNN", "cnn", "raw", False),
    ("BiLSTM", "bilstm", "raw", False),
    ("CNN-BiLSTM", "cnn_bilstm", "raw", False),
    ("TwinShield-Fusion", "twinshield", "fusion", True),
]


def zd_path(name: str) -> str:
    return os.path.join(SAVED, f"zeroday__{name}")


# ---------------------------------------------------------------------------
# Contamination control
# ---------------------------------------------------------------------------
def clean_masks(p: Prepared, id_hist: np.ndarray, unknown: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Which frames may be used for fitting once `unknown` is quarantined.

    Dropping the unknown class from the *label* column is not enough.  A frame
    labelled Normal whose 32-frame context window contains injected brake frames
    has still shown the detector what that attack looks like, and a Normal frame
    whose per-ECU history contains them has still shown the Digital Twin.  Both
    are excluded, so "the model never saw this attack" is true of the inputs and
    not merely of the labels.
    """
    y = p.windows.y
    is_unknown = (y == unknown)
    window_clean = ~is_unknown[p.windows.gather_idx].any(axis=1)
    history_clean = ~is_unknown[id_hist].any(axis=1) & ~is_unknown
    return window_clean, history_clean


# ---------------------------------------------------------------------------
# Detector training on the reduced label space
# ---------------------------------------------------------------------------
def build_model(key: str, n_features: int, n_classes: int, n_streams: int):
    if key == "cnn":
        return CNNDetector(n_features, n_classes)
    if key == "bilstm":
        return BiLSTMDetector(n_features, n_classes)
    if key == "cnn_bilstm":
        return CNNBiLSTMDetector(n_features, n_classes)
    if key == "twinshield":
        return TwinShieldFusion(n_features, n_classes, n_streams)
    raise ValueError(key)


def _weights(y: np.ndarray, idx: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y[idx], minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_detector(model, X, gather, y_small, tr_idx, va_idx, n_classes,
                   streams=None, mask=None, epochs: int = 6, batch: int = 512,
                   lr: float = 1e-3, tag: str = ""):
    fusion = streams is not None
    kw = dict(streams=streams, mask=mask) if fusion else {}
    dl_tr = DataLoader(SeqDataset(X, gather, y_small, tr_idx, **kw),
                       batch_size=batch, shuffle=True)
    dl_va = DataLoader(SeqDataset(X, gather, y_small, va_idx, **kw),
                       batch_size=2048, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=_weights(y_small, tr_idx, n_classes))

    best, best_state = -1.0, None
    for ep in range(epochs):
        t0 = time.time()
        trl, _, _ = _run_epoch(model, dl_tr, lossf, opt, fusion)
        vll, vp, vt = _run_epoch(model, dl_va, lossf, None, fusion)
        vf1 = f1_score(vt, vp, average="macro", labels=range(n_classes), zero_division=0)
        if vf1 > best:
            best = vf1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    [{tag}] epoch {ep+1}/{epochs}  train {trl:.4f}  val {vll:.4f}"
              f"  val macroF1 {vf1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best)


@torch.no_grad()
def predict(model, X, gather, y_small, idx, streams=None, mask=None,
            batch: int = 2048) -> np.ndarray:
    model.eval()
    fusion = streams is not None
    kw = dict(streams=streams, mask=mask) if fusion else {}
    dl = DataLoader(SeqDataset(X, gather, y_small, idx, **kw), batch_size=batch,
                    shuffle=False)
    out = []
    for b in dl:
        if fusion:
            x, st, mk, _ = [t.to(DEVICE) for t in b]
            logits = model(x, st, mk)
        else:
            x, _ = [t.to(DEVICE) for t in b]
            logits = model(x)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Leave-one-attack-out (zero-day) experiment on GEM-CAN")
    ap.add_argument("--unknown", default="Brake Tampering", choices=CLASS_NAMES[1:],
                    help="attack class removed from every training stage")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--dt-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="false-positive rate the anomaly threshold is set to, "
                         "measured on held-out NORMAL validation frames")
    ap.add_argument("--use-id", dest="use_id", action="store_true", default=True)
    ap.add_argument("--no-use-id", dest="use_id", action="store_false",
                    help="drop the arbitration ID from detector inputs")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(SAVED, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    unknown_name = args.unknown
    unknown = CLASS_TO_IDX[unknown_name]
    known_names = [c for c in CLASS_NAMES if c != unknown_name]
    remap = {CLASS_TO_IDX[c]: i for i, c in enumerate(known_names)}
    n_known = len(known_names)

    print("=" * 74)
    print(f"TwinShield-Fusion :: zero-day experiment -- '{unknown_name}' held out")
    print("=" * 74)

    print("\n[1/6] Preparing dataset ...", flush=True)
    p = prepare()
    y = p.windows.y
    split = p.windows.split
    gather = p.windows.gather_idx
    id_index = np.ascontiguousarray(
        p.df["arbitration_id"].map(lambda v: p.id_vocab.get(int(v), 0)).to_numpy(),
        dtype=np.int64)
    n_streams = len(p.id_vocab) + 1
    if not (y == unknown).any():
        raise SystemExit(f"'{unknown_name}' does not occur in this dataset.")

    print("\n[2/6] Quarantining the unknown class ...", flush=True)
    id_hist = build_id_history(p.df, p.windows.block_id)
    window_clean, history_clean = clean_masks(p, id_hist, unknown)

    is_train, is_val, is_test = (split == "train"), (split == "val"), (split == "test")
    tr_idx = np.where(is_train & (y != unknown) & window_clean)[0]
    va_idx = np.where(is_val & (y != unknown) & window_clean)[0]
    te_idx = np.where(is_test)[0]
    dropped_tr = int((is_train & (y != unknown) & ~window_clean).sum())
    dropped_va = int((is_val & (y != unknown) & ~window_clean).sum())
    print(f"      train {len(tr_idx):,}  val {len(va_idx):,}  test {len(te_idx):,} "
          f"(test untouched, still holds {int((y[te_idx] == unknown).sum()):,} "
          f"{unknown_name} frames)")
    print(f"      additionally dropped {dropped_tr:,} train / {dropped_va:,} val frames "
          f"whose context window contained the unknown class")

    # Reduced label space: the unknown class has no index the models can emit.
    y_small = np.full(len(y), -1, dtype=np.int64)
    for orig, new in remap.items():
        y_small[y == orig] = new

    print("\n[3/6] Training the Digital Twin (normal traffic, unknown class removed) ...",
          flush=True)
    print(f"      fit on {int((is_train & (y == 0) & history_clean).sum()):,} normal "
          f"frames with a clean per-ECU history")

    # train_digital_twin() selects its own fit/val indices from the split, so the
    # quarantine is applied by handing it a view whose split column marks every
    # contaminated frame as unusable.  Cheaper and less error-prone than
    # duplicating the training loop here.
    masked_split = split.copy()
    masked_split[(y == 0) & ~history_clean] = "excl"
    orig_split = p.windows.split
    p.windows.split = masked_split
    try:
        dt, dt_metrics = train_digital_twin(p, id_hist, id_index,
                                            epochs=args.dt_epochs, batch=args.batch)
    finally:
        p.windows.split = orig_split
    torch.save(dt.state_dict(), zd_path("digital_twin.pt"))

    payload = p.df[DATA_COLS].to_numpy(dtype=np.float32) / 255.0
    residual = compute_residuals(dt, payload, id_hist, id_index)
    expected = predict_payload(dt, payload, id_hist, id_index)
    np.save(zd_path("residuals.npy"), residual)
    np.save(zd_path("expected_payload.npy"), expected)

    print("\n[4/6] Building ECU residual streams and fitting the anomaly gate ...",
          flush=True)
    streams, mask = build_ecu_streams(residual, gather, id_index, n_streams)
    np.savez(zd_path("ecu_streams.npz"), streams=streams, mask=mask)

    gate = ResidualAnomalyGate.fit(
        residual, id_index, streams, mask,
        fit_idx=np.where(is_train & (y == 0) & window_clean)[0],
        cal_idx=np.where(is_val & (y == 0) & window_clean)[0],
        target_fpr=args.target_fpr)
    gate.save(zd_path("gate.npz"))
    print(f"      threshold z = {gate.threshold:.3f} at a {args.target_fpr*100:.1f}% "
          f"target FPR on {gate.n_cal_frames:,} held-out normal frames")

    gate_alarm_all = gate.alarm(residual, id_index, streams, mask)
    score_test = gate.score(residual[te_idx], id_index[te_idx],
                            streams[te_idx], mask[te_idx])
    y_test = y[te_idx]

    print("\n[5/6] Training detectors on the reduced label space ...", flush=True)
    base_feat = p.features(args.use_id)
    X_raw = base_feat
    X_fusion = np.concatenate([base_feat, residual], axis=1).astype(np.float32)

    models_out: Dict[str, Dict] = {}
    for name, key, mode, uses_gate in ZD_MODELS:
        set_seed(args.seed)
        X = X_fusion if mode == "fusion" else X_raw
        fusion = key == "twinshield"
        model = build_model(key, X.shape[1], n_known, n_streams)
        t0 = time.time()
        model, best_val = train_detector(
            model, X, gather, y_small, tr_idx, va_idx, n_known,
            streams=streams if fusion else None, mask=mask if fusion else None,
            epochs=args.epochs, batch=args.batch, tag=name)
        torch.save(model.state_dict(), zd_path(f"{key}.pt"))

        proba = predict(model, X, gather, y_small, te_idx,
                        streams=streams if fusion else None,
                        mask=mask if fusion else None)
        pred = proba.argmax(1)

        # The comparable question across models with different label spaces:
        # did this frame raise an alarm at all?
        alarm_sup = pred != 0                       # index 0 is Normal in both spaces
        gate_alarm = gate_alarm_all[te_idx]
        alarm = (alarm_sup | gate_alarm) if uses_gate else alarm_sup

        per_class = {}
        for c, cname in enumerate(CLASS_NAMES):
            sel = y_test == c
            if not sel.any():
                continue
            per_class[cname] = {
                "support": int(sel.sum()),
                "alarm_rate": float(alarm[sel].mean()),
                "supervised_alarm_rate": float(alarm_sup[sel].mean()),
                "gate_alarm_rate": float(gate_alarm[sel].mean()),
                # What the supervised head actually called these frames.
                "predicted_as": {known_names[k]: int((pred[sel] == k).sum())
                                 for k in range(n_known) if (pred[sel] == k).any()},
            }

        # Closed-set quality on the classes this model *was* trained on, so a
        # model that simply alarms on everything cannot look good here.
        known_sel = y_test != unknown
        m_known = evaluate(y_small[te_idx][known_sel], pred[known_sel], known_names)

        models_out[name] = {
            "arch": key, "feature_mode": mode, "uses_anomaly_gate": uses_gate,
            "n_features": int(X.shape[1]),
            "best_val_macro_f1": best_val,
            "train_seconds": round(time.time() - t0, 1),
            "known_class_metrics": m_known,
            "known_per_class_f1": per_class_f1(y_small[te_idx][known_sel],
                                               pred[known_sel], known_names),
            "per_class": per_class,
            "unknown_detection_rate": per_class.get(unknown_name, {}).get("alarm_rate", 0.0),
            "false_positive_rate": per_class.get("Normal", {}).get("alarm_rate", 0.0),
        }
        u = models_out[name]["unknown_detection_rate"]
        print(f"    -> {name}: unknown '{unknown_name}' detected "
              f"{u*100:.1f}%  ·  normal FPR {models_out[name]['false_positive_rate']*100:.2f}%"
              f"  ·  known macro-F1 {m_known['Macro F1']:.4f}")

    print("\n[6/6] Characterising the anomaly gate ...", flush=True)
    normal_sel = y_test == 0
    groups = {c: (y_test == CLASS_TO_IDX[c]) for c in CLASS_NAMES[1:]
              if (y_test == CLASS_TO_IDX[c]).any()}
    gate_sweep = sweep(score_test, normal_sel, groups)

    auc: Dict[str, float] = {}
    for cname, sel in groups.items():
        both = normal_sel | sel
        if sel.sum() and normal_sel.sum():
            auc[cname] = float(roc_auc_score(sel[both].astype(int), score_test[both]))

    ex = gate.explain(residual[te_idx], id_index[te_idx], streams[te_idx], mask[te_idx])
    score_stats = {}
    for c, cname in enumerate(CLASS_NAMES):
        sel = y_test == c
        if sel.any():
            s = score_test[sel]
            fired = sel & (score_test > gate.threshold)
            score_stats[cname] = {
                "mean": float(s.mean()), "median": float(np.median(s)),
                "support": int(sel.sum()),
                # Quantiles rather than raw scores: enough for the Results page to
                # draw the spread without shipping one number per test frame.
                **{f"p{q}": float(np.quantile(s, q / 100))
                   for q in (5, 25, 50, 75, 95)},
                "min": float(s.min()), "max": float(s.max()),
                "detected": float((s > gate.threshold).mean()),
                # Which of the gate's two views carried the alarm.
                "channel_frame_share": (float((ex["channel"][fired] == "frame").mean())
                                        if fired.any() else 0.0),
                "mean_frame_z": float(ex["frame_z"][sel].mean()),
                "mean_ecu_z": float(ex["ecu_z"][sel].mean()),
            }

    # Per-CAN-ID residual, the physical reason the gate fires (or does not).
    hexes = p.df["arbitration_id_hex"].to_numpy()
    id_table = []
    for cname in CLASS_NAMES[1:]:
        sel_a = (y == CLASS_TO_IDX[cname])
        if not sel_a.any():
            continue
        for h in np.unique(hexes[sel_a]):
            on_id = hexes == h
            n_sel, a_sel = on_id & (y == 0), on_id & sel_a
            if not (n_sel.any() and a_sel.any()):
                continue
            id_table.append({
                "can_id": str(h), "class": cname,
                "mean_residual_normal": float(residual[n_sel].mean() * 255),
                "mean_residual_attack": float(residual[a_sel].mean() * 255),
                "n_normal": int(n_sel.sum()), "n_attack": int(a_sel.sum()),
            })

    payload_out = {
        "unknown_class": unknown_name,
        "known_classes": known_names,
        "class_names": CLASS_NAMES,
        "use_id": bool(args.use_id),
        "digital_twin": dt_metrics,
        "gate": {
            **gate.describe(),
            "sweep": gate_sweep,
            "auc_vs_normal": auc,
            "score_stats": score_stats,
            "residual_by_can_id": id_table,
        },
        "models": models_out,
        "model_order": [m[0] for m in ZD_MODELS],
        "split_sizes": {"train_used": int(len(tr_idx)), "val_used": int(len(va_idx)),
                        "test": int(len(te_idx)),
                        "dropped_contaminated_train": dropped_tr,
                        "dropped_contaminated_val": dropped_va},
        "test_class_support": {CLASS_NAMES[c]: int((y_test == c).sum())
                               for c in range(len(CLASS_NAMES))},
        "config": {"epochs": args.epochs, "dt_epochs": args.dt_epochs,
                   "batch": args.batch, "seed": args.seed,
                   "target_fpr": args.target_fpr,
                   "window": int(gather.shape[1]), "n_streams": n_streams},
    }
    with open(ZD_FILE, "w") as fh:
        json.dump(payload_out, fh, indent=2)

    print("\n" + "=" * 74)
    prop = models_out.get("TwinShield-Fusion", {})
    base = [models_out[m]["unknown_detection_rate"] for m in
            ("CNN", "BiLSTM", "CNN-BiLSTM") if m in models_out]
    print(f"Unknown attack '{unknown_name}' on the held-out test split:")
    for name, _, _, _ in ZD_MODELS:
        if name in models_out:
            print(f"   {name:<20} {models_out[name]['unknown_detection_rate']*100:6.2f}% detected")
    if base and prop:
        print(f"\nBest baseline {max(base)*100:.2f}%  vs  TwinShield-Fusion "
              f"{prop['unknown_detection_rate']*100:.2f}%")
    print(f"\nResults -> {ZD_FILE}")
    print("=" * 74)


if __name__ == "__main__":
    main()
