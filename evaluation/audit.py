"""
Dataset audit for research validity.

Answers, with numbers rather than assertions:
  * how imbalanced are the classes?
  * are there duplicate frames, and do duplicates straddle train/test?
  * do normal and attack traffic share arbitration IDs?
  * can CAN ID *alone* identify an attack (the classic CAN-IDS shortcut)?

The CAN-ID shortcut result is the reason this project also trains and reports
an ID-free variant of every model.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from preprocessing.loader import CLASS_NAMES, DATA_COLS
from preprocessing.prepare import Prepared


def class_balance(p: Prepared) -> pd.DataFrame:
    y = p.windows.y
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    total = counts.sum()
    return pd.DataFrame({
        "Class": CLASS_NAMES,
        "Frames": counts,
        "Share %": (counts / total * 100).round(3),
        "Imbalance ratio vs largest": (counts.max() / np.maximum(counts, 1)).round(1),
    })


def duplicate_report(p: Prepared) -> Dict[str, object]:
    """Exact duplicate frames on (id, dlc, payload) and whether they cross splits."""
    df = p.df
    key_cols = ["arbitration_id", "dlc"] + DATA_COLS

    # Exact grouping on the integer columns.  A string join over 143k rows takes
    # tens of seconds and would stall the Results page on first load; np.unique
    # over the integer matrix is exact and about two orders of magnitude faster.
    vals = np.ascontiguousarray(df[key_cols].to_numpy(dtype=np.int64))
    _, key = np.unique(vals, axis=0, return_inverse=True)
    key = key.ravel()

    n_total = len(key)
    n_unique = int(key.max()) + 1 if n_total else 0

    split = p.windows.split
    tr_keys = np.unique(key[split == "train"])
    te_keys = key[split == "test"]

    # A payload group straddles splits if it appears under more than one split.
    order = np.lexsort((split, key))
    k_sorted, s_sorted = key[order], split[order]
    new_group = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    new_pair = np.r_[True, (k_sorted[1:] != k_sorted[:-1]) | (s_sorted[1:] != s_sorted[:-1])]
    pairs_per_group = np.add.reduceat(new_pair.astype(np.int64),
                                      np.flatnonzero(new_group)) if n_total else np.array([])
    crossing = int((pairs_per_group > 1).sum())

    test_in_train = float(np.isin(te_keys, tr_keys).mean() * 100) if len(te_keys) else 0.0

    return {
        "total_frames": int(n_total),
        "unique_frames": int(n_unique),
        "duplicate_frames": int(n_total - n_unique),
        "duplicate_share_pct": round((n_total - n_unique) / n_total * 100, 2),
        "distinct_payloads_crossing_splits": crossing,
        "test_frames_with_identical_twin_in_train_pct": round(test_in_train, 2),
    }


def can_id_overlap(p: Prepared) -> pd.DataFrame:
    """Which arbitration IDs carry which classes."""
    df = p.df.copy()
    df["class_name"] = pd.Categorical(df["class_name"], categories=CLASS_NAMES)
    tab = pd.crosstab(df["arbitration_id_hex"], df["class_name"])
    tab["Total"] = tab.sum(axis=1)
    tab["Attack frames"] = tab[[c for c in CLASS_NAMES if c != "Normal"]].sum(axis=1)
    tab["Shared normal+attack"] = np.where(
        (tab["Normal"] > 0) & (tab["Attack frames"] > 0), "YES", "no"
    )
    return tab.sort_values("Total", ascending=False).reset_index()


def can_id_shortcut(p: Prepared) -> Dict[str, object]:
    """Majority-class lookup on arbitration ID only, fitted on train, scored on test.

    If this scores highly, CAN ID alone is a shortcut and any model given the ID
    can appear excellent without learning anything about payload behaviour.
    """
    df = p.df
    ids = df["arbitration_id"].to_numpy()
    y = p.windows.y
    split = p.windows.split

    tr, te = split == "train", split == "test"
    lookup: Dict[int, int] = {}
    for cid in np.unique(ids[tr]):
        vals = y[tr][ids[tr] == cid]
        lookup[int(cid)] = int(np.bincount(vals, minlength=len(CLASS_NAMES)).argmax())

    majority = int(np.bincount(y[tr], minlength=len(CLASS_NAMES)).argmax())
    pred = np.array([lookup.get(int(c), majority) for c in ids[te]])

    per_class = f1_score(y[te], pred, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
    return {
        "accuracy": float((pred == y[te]).mean()),
        "macro_f1": float(f1_score(y[te], pred, average="macro", zero_division=0)),
        "per_class_f1": {CLASS_NAMES[i]: float(per_class[i]) for i in range(len(CLASS_NAMES))},
        "id_to_class": {f"{k:08X}": CLASS_NAMES[v] for k, v in sorted(lookup.items())},
    }


def run_full_audit(p: Prepared) -> Dict[str, object]:
    return {
        "class_balance": class_balance(p),
        "duplicates": duplicate_report(p),
        "can_id_overlap": can_id_overlap(p),
        "can_id_shortcut": can_id_shortcut(p),
    }
