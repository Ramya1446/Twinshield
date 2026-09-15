"""
TwinShield-Fusion -- full training pipeline.

Trains, on one consistent leakage-controlled split of the real GEM-CAN data:

    1. GRU Digital Twin            (fitted on Normal training frames only)
    2. CNN                baseline (raw CAN)
    3. BiLSTM             baseline (raw CAN)
    4. CNN-BiLSTM         baseline (raw CAN)
    5. + Digital Twin              (raw + scalar residual magnitude)
    6. + Residual                  (raw + per-byte residual vector)
    7. TwinShield-Fusion           (raw + residual + adaptive multi-ECU fusion)

Everything is run twice: once with CAN ID available to the model and once with
CAN ID removed, because the audit shows arbitration ID alone already separates
DoS perfectly.  Results go to results/ and weights to saved_models/.

    python train.py                  # full run
    python train.py --epochs 3       # quick run
    python train.py --variants no_id  # only the ID-free controlled experiment
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import evaluate, per_class_f1
from models.bilstm import BiLSTMDetector
from models.cnn import CNNDetector
from models.cnn_bilstm import CNNBiLSTMDetector
from models.digital_twin import GRUDigitalTwin, compute_residuals, predict_payload
from models.residual_fusion import TwinShieldFusion
from preprocessing.loader import CLASS_NAMES, DATA_COLS
from preprocessing.prepare import Prepared, prepare
from preprocessing.windowing import (build_ecu_streams, build_id_history,
                                     rank_sort_streams)

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(ROOT, "saved_models")
RESULTS = os.path.join(ROOT, "results")
DEVICE = "cpu"

# Model registry: display name -> (builder key, feature mode)
MODEL_SPECS: List[Tuple[str, str, str]] = [
    ("CNN", "cnn", "raw"),
    ("BiLSTM", "bilstm", "raw"),
    ("CNN-BiLSTM", "cnn_bilstm", "raw"),
    ("GRU Digital Twin + CNN-BiLSTM", "cnn_bilstm", "residual"),
    ("TwinShield-Fusion", "twinshield", "fusion"),
]

# Ablation ladder (the key research visual)
ABLATION_SPECS: List[Tuple[str, str, str]] = [
    ("CNN-BiLSTM", "cnn_bilstm", "raw"),
    ("+ Digital Twin", "cnn_bilstm", "dt_scalar"),
    ("+ Residual", "cnn_bilstm", "residual"),
    ("+ Adaptive Fusion", "twinshield", "fusion"),
]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SeqDataset(Dataset):
    """Gathers a [WINDOW, F] context window per sample without materialising all."""

    def __init__(self, X: np.ndarray, gather: np.ndarray, y: np.ndarray,
                 idx: np.ndarray, streams: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None):
        self.X, self.gather, self.y, self.idx = X, gather, y, idx
        self.streams, self.mask = streams, mask

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        x = torch.from_numpy(self.X[self.gather[j]])
        y = torch.tensor(self.y[j], dtype=torch.long)
        if self.streams is None:
            return x, y
        return x, torch.from_numpy(self.streams[j]), torch.from_numpy(self.mask[j]), y


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------
def train_digital_twin(p: Prepared, id_hist: np.ndarray, id_index: np.ndarray,
                       epochs: int = 6, batch: int = 512, lr: float = 2e-3,
                       verbose: bool = True) -> Tuple[GRUDigitalTwin, Dict[str, float]]:
    payload = p.df[DATA_COLS].to_numpy(dtype=np.float32) / 255.0
    y, split = p.windows.y, p.windows.split

    fit_idx = np.where((split == "train") & (y == 0))[0]
    val_idx = np.where((split == "val") & (y == 0))[0]

    model = GRUDigitalTwin(n_ids=int(id_index.max()) + 1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    payload_t = torch.from_numpy(payload)

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(fit_idx)
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            xb = payload_t[id_hist[b]].to(DEVICE)
            ib = torch.from_numpy(id_index[b]).long().to(DEVICE)
            tb = payload_t[b].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb, ib), tb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if verbose:
            print(f"    DT epoch {ep+1}/{epochs}  train MSE {tot/max(nb,1):.6f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    res_val = compute_residuals(model, payload, id_hist, id_index)[val_idx]
    mae = float(np.mean(res_val) * 255.0)
    mse = float(np.mean((res_val * 255.0) ** 2))
    metrics = {"MAE": mae, "RMSE": float(np.sqrt(mse)), "MSE": mse,
               "MAE_normalised": float(np.mean(res_val)),
               "n_val_normal_frames": int(len(val_idx))}
    if verbose:
        print(f"    DT held-out normal frames: MAE {mae:.3f} bytes  RMSE {metrics['RMSE']:.3f}")
    return model, metrics


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------
def _class_weights(y: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y[idx], minlength=len(CLASS_NAMES)).astype(np.float64)
    w = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def _run_epoch(model, loader, lossf, opt=None, fusion=False):
    training = opt is not None
    model.train() if training else model.eval()
    tot, n = 0.0, 0
    preds, trues = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if fusion:
                x, st, mk, y = [b.to(DEVICE) for b in batch]
                logits = model(x, st, mk)
            else:
                x, y = [b.to(DEVICE) for b in batch]
                logits = model(x)
            loss = lossf(logits, y)
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
    return tot / max(n, 1), np.concatenate(preds), np.concatenate(trues)


def train_classifier(model, p: Prepared, X: np.ndarray, streams, mask,
                     epochs: int, batch: int = 512, lr: float = 1e-3,
                     fusion: bool = False, tag: str = "", verbose: bool = True):
    g, y = p.windows.gather_idx, p.windows.y
    tr, va = p.idx("train"), p.idx("val")

    kw = dict(streams=streams, mask=mask) if fusion else {}
    dl_tr = DataLoader(SeqDataset(X, g, y, tr, **kw), batch_size=batch, shuffle=True)
    dl_va = DataLoader(SeqDataset(X, g, y, va, **kw), batch_size=2048, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=_class_weights(y, tr))

    best, best_state = -1.0, None
    for ep in range(epochs):
        t0 = time.time()
        trl, _, _ = _run_epoch(model, dl_tr, lossf, opt, fusion)
        vll, vp, vt = _run_epoch(model, dl_va, lossf, None, fusion)
        vf1 = f1_score(vt, vp, average="macro", labels=range(len(CLASS_NAMES)),
                       zero_division=0)
        if vf1 > best:
            best = vf1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    [{tag}] epoch {ep+1}/{epochs}  train {trl:.4f}  val {vll:.4f}"
                  f"  val macroF1 {vf1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best)


@torch.no_grad()
def predict_proba(model, p: Prepared, X, streams, mask, idx: np.ndarray,
                  fusion: bool = False, batch: int = 2048) -> np.ndarray:
    model.eval()
    g, y = p.windows.gather_idx, p.windows.y
    kw = dict(streams=streams, mask=mask) if fusion else {}
    dl = DataLoader(SeqDataset(X, g, y, idx, **kw), batch_size=batch, shuffle=False)
    out = []
    for b in dl:
        if fusion:
            x, st, mkb, _ = [t.to(DEVICE) for t in b]
            logits = model(x, st, mkb)
        else:
            x, _ = [t.to(DEVICE) for t in b]
            logits = model(x)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_attention(model: TwinShieldFusion, streams, mask, idx: np.ndarray,
                      batch: int = 4096) -> np.ndarray:
    """Real attention weights over ECU streams for the given frames."""
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        st = torch.from_numpy(streams[b]).to(DEVICE)
        mk = torch.from_numpy(mask[b]).to(DEVICE)
        _, alpha = model.fusion(st, mk)
        out.append(alpha.cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def build_features(p: Prepared, use_id: bool, residual: np.ndarray, mode: str) -> np.ndarray:
    base = p.features(use_id)
    if mode == "raw":
        return base
    if mode == "dt_scalar":
        return np.concatenate([base, residual.mean(axis=1, keepdims=True)], axis=1).astype(np.float32)
    if mode in ("residual", "fusion"):
        return np.concatenate([base, residual], axis=1).astype(np.float32)
    raise ValueError(mode)


def build_model(key: str, n_features: int, n_streams: int):
    n_cls = len(CLASS_NAMES)
    if key == "cnn":
        return CNNDetector(n_features, n_cls)
    if key == "bilstm":
        return BiLSTMDetector(n_features, n_cls)
    if key == "cnn_bilstm":
        return CNNBiLSTMDetector(n_features, n_cls)
    if key == "twinshield":
        return TwinShieldFusion(n_features, n_cls, n_streams)
    raise ValueError(key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train TwinShield-Fusion on GEM-CAN")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--dt-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--variants", nargs="+", default=["with_id", "no_id"],
                    choices=["with_id", "no_id"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["cnn", "bilstm", "cnn_bilstm", "twinshield"],
                    help="retrain only these architectures and merge the rest")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(SAVED, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 74)
    print("TwinShield-Fusion :: training on real GEM-CAN data")
    print("=" * 74)

    print("\n[1/5] Preparing dataset ...", flush=True)
    p = prepare()
    id_index = np.ascontiguousarray(
        p.df["arbitration_id"].map(lambda v: p.id_vocab.get(int(v), 0)).to_numpy(),
        dtype=np.int64,
    )
    n_streams = len(p.id_vocab) + 1
    print(f"      frames={len(p.df):,}  streams={n_streams}  "
          f"train/val/test={len(p.idx('train')):,}/{len(p.idx('val')):,}/{len(p.idx('test')):,}")

    print("\n[2/5] Building per-ECU history index ...", flush=True)
    id_hist = build_id_history(p.df, p.windows.block_id)

    dt_path = os.path.join(SAVED, "digital_twin.pt")
    res_path = os.path.join(SAVED, "residuals.npy")
    reuse_twin = (args.only is not None and os.path.exists(dt_path)
                  and os.path.exists(res_path))

    payload = p.df[DATA_COLS].to_numpy(dtype=np.float32) / 255.0
    dt_metrics: Dict[str, float] = {}

    if reuse_twin:
        # Retraining a subset of detectors must not change the residuals the
        # other detectors were fitted on, so the existing twin is reused.
        print("\n[3/5] Reusing existing GRU Digital Twin (--only given) ...", flush=True)
        dt = GRUDigitalTwin(n_ids=int(id_index.max()) + 1)
        dt.load_state_dict(torch.load(dt_path, map_location=DEVICE))
        dt.eval()
        residual = np.load(res_path)
        expected = np.load(os.path.join(SAVED, "expected_payload.npy"))
    else:
        print("\n[3/5] Training GRU Digital Twin (normal traffic only) ...", flush=True)
        dt, dt_metrics = train_digital_twin(p, id_hist, id_index,
                                            epochs=args.dt_epochs, batch=args.batch)
        torch.save(dt.state_dict(), dt_path)
        residual = compute_residuals(dt, payload, id_hist, id_index)
        expected = predict_payload(dt, payload, id_hist, id_index)
        np.save(res_path, residual)
        np.save(os.path.join(SAVED, "expected_payload.npy"), expected)
        np.save(os.path.join(SAVED, "id_index.npy"), id_index)

    print("\n[4/5] Building adaptive ECU residual streams ...", flush=True)
    streams, mask = build_ecu_streams(residual, p.windows.gather_idx, id_index, n_streams)

    test_idx = p.idx("test")
    y_test = p.windows.y[test_idx]

    all_results: Dict[str, Dict] = {}
    print("\n[5/5] Training detectors ...", flush=True)

    for variant in args.variants:
        use_id = variant == "with_id"
        print(f"\n  --- variant: {variant} (CAN ID {'included' if use_id else 'EXCLUDED'}) ---")
        variant_results: Dict[str, Dict] = {}

        # In the ID-free experiment the fusion slots are rank-ordered, otherwise
        # a fixed slot-per-ID layout would hand the arbitration ID back to the
        # model through the stream index alone.
        if use_id:
            v_streams, v_mask = streams, mask
        else:
            v_streams, v_mask = rank_sort_streams(streams, mask)
            print("      fusion streams rank-sorted (slot index carries no CAN ID)")

        # Several display names share an identical (architecture, feature-mode)
        # configuration -- e.g. "GRU Digital Twin + CNN-BiLSTM" in the comparison
        # table is the same trained model as "+ Residual" in the ablation ladder.
        # Train each unique configuration once and map the display names onto it.
        display_map: Dict[str, Tuple[str, str]] = {}
        for name, key, mode in list(MODEL_SPECS) + list(ABLATION_SPECS):
            display_map.setdefault(name, (key, mode))

        trained: Dict[Tuple[str, str], Dict] = {}
        configs = [c for c in dict.fromkeys(display_map.values())
                   if args.only is None or c[0] in args.only]
        if args.only:
            print(f"      retraining only: {', '.join(args.only)}")
        for cfg in configs:
            key, mode = cfg
            label = next(n for n, c in display_map.items() if c == cfg)
            set_seed(args.seed)
            X = build_features(p, use_id, residual, mode)
            fusion = key == "twinshield"
            model = build_model(key, X.shape[1], n_streams)
            t0 = time.time()
            model, best_val = train_classifier(
                model, p, X, v_streams if fusion else None, v_mask if fusion else None,
                epochs=args.epochs, batch=args.batch, fusion=fusion,
                tag=f"{variant}/{label}")

            proba = predict_proba(model, p, X, v_streams if fusion else None,
                                  v_mask if fusion else None, test_idx, fusion=fusion)
            pred = proba.argmax(1)
            m = evaluate(y_test, pred)
            trained[cfg] = {
                "metrics": m,
                "per_class_f1": per_class_f1(y_test, pred),
                "best_val_macro_f1": best_val,
                "n_features": int(X.shape[1]),
                "train_seconds": round(time.time() - t0, 1),
                "arch": key, "feature_mode": mode,
            }
            print(f"    -> {label}: test macroF1 {m['Macro F1']:.4f}  acc {m['Accuracy']:.4f}")

            slug = f"{key}__{mode}"
            torch.save(model.state_dict(), os.path.join(SAVED, f"{variant}__{slug}.pt"))

            if fusion:
                np.save(os.path.join(SAVED, f"{variant}__twinshield_test_proba.npy"), proba)
                alpha = predict_attention(model, v_streams, v_mask, test_idx)
                np.save(os.path.join(SAVED, f"{variant}__twinshield_test_attention.npy"), alpha)

        for name, cfg in display_map.items():
            if cfg in trained:
                variant_results[name] = trained[cfg]

        all_results[variant] = variant_results

    payload_out = {
        "class_names": CLASS_NAMES,
        "digital_twin": dt_metrics,
        "results": all_results,
        "config": {
            "epochs": args.epochs, "dt_epochs": args.dt_epochs, "batch": args.batch,
            "seed": args.seed, "window": int(p.windows.gather_idx.shape[1]),
            "n_streams": n_streams,
        },
        "split_sizes": {s: int(len(p.idx(s))) for s in ("train", "val", "test")},
        "test_class_support": {CLASS_NAMES[i]: int((y_test == i).sum())
                               for i in range(len(CLASS_NAMES))},
        "model_order": [s[0] for s in MODEL_SPECS],
        "ablation_order": [s[0] for s in ABLATION_SPECS],
    }
    # Merge with any previously trained variants so re-running a single
    # variant does not discard the others.
    out_path = os.path.join(RESULTS, "results.json")
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                prev = json.load(fh)
            merged = {v: dict(m) for v, m in prev.get("results", {}).items()}
            for v, models in payload_out["results"].items():
                merged.setdefault(v, {}).update(models)
            payload_out["results"] = merged
            # keep Digital Twin metrics if this run did not retrain it
            if not payload_out.get("digital_twin") and prev.get("digital_twin"):
                payload_out["digital_twin"] = prev["digital_twin"]
        except (OSError, ValueError):
            pass
    with open(out_path, "w") as fh:
        json.dump(payload_out, fh, indent=2)

    print("\n" + "=" * 74)
    print("Done. Results -> results/results.json ; weights -> saved_models/")
    print("=" * 74)


if __name__ == "__main__":
    main()
