"""Single entry point that turns the raw GEM-CAN CSVs into model-ready arrays."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import CLASS_NAMES, load_all
from .preprocessing import (
    add_temporal_features,
    build_id_vocabulary,
    feature_names,
    frame_feature_matrix,
)
from .windowing import WINDOW, SplitData, assign_blocks, build_windows, split_blocks

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models"
)
CACHE_FILE = os.path.join(CACHE_DIR, "prepared_dataset.pkl")


@dataclass
class Prepared:
    df: pd.DataFrame
    X_id: np.ndarray            # [N, F_with_id]
    X_noid: np.ndarray          # [N, F_no_id]
    windows: SplitData
    id_vocab: Dict[int, int]
    names_id: List[str]
    names_noid: List[str]

    def idx(self, split: str) -> np.ndarray:
        return np.where(self.windows.split == split)[0]

    def features(self, use_id: bool) -> np.ndarray:
        return self.X_id if use_id else self.X_noid

    def names(self, use_id: bool) -> List[str]:
        return self.names_id if use_id else self.names_noid


def prepare(path: Optional[str] = None, use_cache: bool = True) -> Prepared:
    if use_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as fh:
            return pickle.load(fh)

    df = load_all(path)
    df = add_temporal_features(df)

    id_vocab = build_id_vocabulary(df)
    X_id = frame_feature_matrix(df, id_vocab, use_id=True)
    X_noid = frame_feature_matrix(df, id_vocab, use_id=False)

    block = assign_blocks(df)
    assignment = split_blocks(df, block)
    windows = build_windows(df, block, assignment)

    prepared = Prepared(
        df=df,
        X_id=X_id,
        X_noid=X_noid,
        windows=windows,
        id_vocab=id_vocab,
        names_id=feature_names(id_vocab, use_id=True),
        names_noid=feature_names(id_vocab, use_id=False),
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(prepared, fh, protocol=4)
    return prepared


def split_report(p: Prepared) -> pd.DataFrame:
    """Per-split class counts -- confirms every class survives the split."""
    rows = []
    for s in ("train", "val", "test"):
        idx = p.idx(s)
        y = p.windows.y[idx]
        row = {"Split": s, "Frames": len(idx)}
        for c, name in enumerate(CLASS_NAMES):
            row[name] = int((y == c).sum())
        rows.append(row)
    return pd.DataFrame(rows)
