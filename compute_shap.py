"""
Compute SHAP attributions for the trained TwinShield-Fusion detector.

Run once after train.py:

    python compute_shap.py

Writes results/shap.npz, which the Results page loads.  Explained frames are
sampled from the held-out test split, stratified across classes so the rare
tampering classes are actually represented.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from explainability.shap_explainer import compute_shap, save
from models.residual_fusion import TwinShieldFusion
from preprocessing.loader import CLASS_NAMES
from preprocessing.prepare import prepare
from preprocessing.windowing import build_ecu_streams

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(ROOT, "saved_models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="with_id", choices=["with_id", "no_id"])
    ap.add_argument("--background", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsamples", type=int, default=64,
                    help="gradient samples per explained frame")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading dataset and Digital Twin outputs ...")
    p = prepare()
    residual = np.load(os.path.join(SAVED, "residuals.npy"))
    id_index = np.ascontiguousarray(
        p.df["arbitration_id"].map(lambda v: p.id_vocab.get(int(v), 0)).to_numpy(),
        dtype=np.int64)
    n_streams = len(p.id_vocab) + 1

    X = np.concatenate([p.features(True), residual], axis=1).astype(np.float32)
    streams, _ = build_ecu_streams(residual, p.windows.gather_idx, id_index, n_streams)

    ckpt = os.path.join(SAVED, f"{args.variant}__twinshield__fusion.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}\nRun python train.py first.")
    model = TwinShieldFusion(X.shape[1], len(CLASS_NAMES), n_streams)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    train_idx, test_idx = p.idx("train"), p.idx("test")
    y = p.windows.y

    background = rng.choice(train_idx, size=min(args.background, len(train_idx)),
                            replace=False)

    # Stratified explain set so rare classes are represented.
    explain = []
    for c in range(len(CLASS_NAMES)):
        pool = test_idx[y[test_idx] == c]
        if len(pool):
            explain.append(rng.choice(pool, size=min(args.per_class, len(pool)),
                                      replace=False))
    explain = np.sort(np.concatenate(explain))
    print(f"Background {len(background)} frames, explaining {len(explain)} test frames",
          flush=True)
    for c in range(len(CLASS_NAMES)):
        print(f"   {CLASS_NAMES[c]:<20} {(y[explain] == c).sum()}")

    print(f"Running shap.GradientExplainer (nsamples={args.nsamples}) ...",
          flush=True)
    payload = compute_shap(
        model, X, p.windows.gather_idx, streams,
        background_idx=background, explain_idx=explain,
        feature_names=p.names(True) + [f"Residual DATA_{i}" for i in range(8)],
        stream_labels=["<unknown>"] + [
            f"{k:08X}" for k, _ in sorted(p.id_vocab.items(), key=lambda kv: kv[1])],
        class_names=CLASS_NAMES,
        nsamples=args.nsamples,
    )
    path = save(payload)
    print(f"Saved -> {path}")
    print(f"feature_shap {payload['feature_shap'].shape}  "
          f"ecu_shap {payload['ecu_shap'].shape}")


if __name__ == "__main__":
    main()
