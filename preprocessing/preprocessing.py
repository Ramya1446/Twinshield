"""
Frame-level feature engineering for GEM-CAN.

Design note (why frame-level, not window-level):
    In the GEM-CAN attack capture, Brake Tampering and Steering Tampering are
    injected as an interleaved pair (brake at t, steering at t+0.56s).  Any
    window wide enough to give temporal context therefore contains *both*
    tampering classes, which makes a single window-level 4-class label
    ill-defined.  We instead classify **each frame** using the W-1 frames of
    context that precede it.  The label is the class of the final frame.

Two explicit feature variants are produced so the CAN-ID shortcut can be
measured rather than assumed:

    with_id : payload(8) + dlc(1) + dt_bus(1) + dt_same_id(1) + id_onehot(N)
    no_id   : payload(8) + dlc(1) + dt_bus(1)

`no_id` deliberately drops dt_same_id as well, because grouping by arbitration
ID to compute it is itself CAN-ID information.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .loader import DATA_COLS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inter-frame timing, computed within each capture source."""
    df = df.copy()

    dt_bus = df.groupby("source")["timestamp"].diff()
    df["dt_bus"] = dt_bus.fillna(0.0).clip(lower=0.0)

    dt_same = df.groupby(["source", "arbitration_id"])["timestamp"].diff()
    # Unseen-before ID -> no previous interval; use 0 as a neutral marker.
    df["dt_same_id"] = dt_same.fillna(0.0).clip(lower=0.0)
    return df


def build_id_vocabulary(df: pd.DataFrame) -> Dict[int, int]:
    """Map arbitration_id -> dense index (index 0 reserved for unknown)."""
    ids = sorted(df["arbitration_id"].unique().tolist())
    return {int(v): i + 1 for i, v in enumerate(ids)}


def frame_feature_matrix(
    df: pd.DataFrame,
    id_vocab: Dict[int, int],
    use_id: bool,
) -> np.ndarray:
    """Return float32 [n_frames, n_features] for the requested variant."""
    n = len(df)

    payload = df[DATA_COLS].to_numpy(dtype=np.float32) / 255.0
    dlc = (df["dlc"].to_numpy(dtype=np.float32) / 8.0).reshape(-1, 1)
    # Timing spans microseconds..seconds -> log compress.
    dt_bus = np.log1p(df["dt_bus"].to_numpy(dtype=np.float32)).reshape(-1, 1)

    blocks: List[np.ndarray] = [payload, dlc, dt_bus]

    if use_id:
        dt_same = np.log1p(df["dt_same_id"].to_numpy(dtype=np.float32)).reshape(-1, 1)
        n_ids = len(id_vocab) + 1
        onehot = np.zeros((n, n_ids), dtype=np.float32)
        idx = df["arbitration_id"].map(lambda v: id_vocab.get(int(v), 0)).to_numpy()
        onehot[np.arange(n), idx] = 1.0
        blocks.extend([dt_same, onehot])

    return np.concatenate(blocks, axis=1).astype(np.float32)


def feature_names(id_vocab: Dict[int, int], use_id: bool) -> List[str]:
    """Human-readable names, used by the SHAP explainer."""
    names = [f"DATA_{i}" for i in range(8)] + ["DLC", "dt_bus"]
    if use_id:
        names.append("dt_same_id")
        inv = {v: k for k, v in id_vocab.items()}
        names.append("CAN ID <unknown>")
        for i in range(1, len(id_vocab) + 1):
            names.append(f"CAN ID {inv[i]:08X}")
    return names


def residual_feature_names() -> List[str]:
    return [f"Residual DATA_{i}" for i in range(8)]
