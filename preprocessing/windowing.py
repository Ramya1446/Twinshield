"""
Context-window construction and leakage-free splitting.

Splitting strategy — contiguous *block* split, stratified by block content
-------------------------------------------------------------------------
Consecutive CAN frames are near-duplicates, so a uniformly random frame split
would leak almost-identical frames between train and test and inflate every
metric.  A purely chronological split is also unusable here: in the GEM-CAN
attack capture the brake/steering tampering is concentrated in the opening
~9% of rows, so a chronological cut puts *zero* tampering frames in the test
set and the rarest classes become unmeasurable.

We therefore:
  1. cut each capture into contiguous blocks of BLOCK_SIZE frames,
  2. label each block by its rarest-present class family
     (tampering > DoS > normal-only),
  3. assign whole blocks to train/val/test, stratified on that family.

A frame's context window is drawn only from frames inside its own block, so no
frame is ever seen by two splits.  Blocks that are adjacent in time can still
be correlated -- that limitation is reported in the audit rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .loader import CLASS_TO_IDX

BLOCK_SIZE = 256
WINDOW = 32
SEED = 42


@dataclass
class SplitData:
    gather_idx: np.ndarray      # [N, WINDOW] int32 -> rows of the frame matrix
    y: np.ndarray               # [N] int64
    split: np.ndarray           # [N] '<U5' train/val/test
    frame_pos: np.ndarray       # [N] int64 -> row of the frame the label belongs to
    block_id: np.ndarray        # [N] int64


def assign_blocks(df: pd.DataFrame) -> np.ndarray:
    """Contiguous block id, restarting at each capture source."""
    block = np.zeros(len(df), dtype=np.int64)
    offset = 0
    for _, idx in df.groupby("source", sort=False).indices.items():
        idx = np.sort(idx)
        local = np.arange(len(idx)) // BLOCK_SIZE
        block[idx] = local + offset
        offset += local.max() + 1
    return block


def _block_family(classes: np.ndarray) -> str:
    s = set(classes.tolist())
    if CLASS_TO_IDX["Brake Tampering"] in s or CLASS_TO_IDX["Steering Tampering"] in s:
        return "tamper"
    if CLASS_TO_IDX["DoS"] in s:
        return "dos"
    return "normal"


def split_blocks(
    df: pd.DataFrame,
    block: np.ndarray,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = SEED,
) -> Dict[int, str]:
    """Stratified assignment of whole blocks to train/val/test."""
    y = df["y"].to_numpy()
    families: Dict[str, list] = {"tamper": [], "dos": [], "normal": []}
    for b in np.unique(block):
        families[_block_family(y[block == b])].append(int(b))

    rng = np.random.default_rng(seed)
    assignment: Dict[int, str] = {}
    for fam, blocks in families.items():
        blocks = np.array(sorted(blocks))
        rng.shuffle(blocks)
        n = len(blocks)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        # Guarantee non-empty val/test for any family that has >= 3 blocks.
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        for i, b in enumerate(blocks):
            assignment[int(b)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assignment


def build_windows(df: pd.DataFrame, block: np.ndarray, assignment: Dict[int, str]) -> SplitData:
    """Left-padded context windows; padding repeats the first frame of the block."""
    n = len(df)
    gather = np.zeros((n, WINDOW), dtype=np.int32)

    positions = np.arange(n, dtype=np.int64)
    # Start row of each frame's block.
    order = np.argsort(block, kind="stable")
    block_sorted = block[order]
    starts = np.zeros(n, dtype=np.int64)
    uniq, first_idx = np.unique(block_sorted, return_index=True)
    block_start = dict(zip(uniq.tolist(), order[first_idx].tolist()))
    for b, s in block_start.items():
        starts[block == b] = s

    for k in range(WINDOW):
        # offset WINDOW-1-k frames back, clamped to the block start
        back = WINDOW - 1 - k
        gather[:, k] = np.maximum(positions - back, starts).astype(np.int32)

    split = np.array([assignment[int(b)] for b in block], dtype="<U5")
    return SplitData(
        gather_idx=gather,
        y=df["y"].to_numpy(dtype=np.int64),
        split=split,
        frame_pos=positions,
        block_id=block,
    )


def gather_windows(X_frames: np.ndarray, gather_idx: np.ndarray) -> np.ndarray:
    """[N, WINDOW] index -> [N, WINDOW, F] float32 tensor."""
    return X_frames[gather_idx]


ID_HISTORY = 8


def build_id_history(df: pd.DataFrame, block: np.ndarray, k: int = ID_HISTORY) -> np.ndarray:
    """[N, k] indices of the k previous frames sharing the same arbitration ID.

    History never crosses a block boundary, so the Digital Twin's context obeys
    the same split isolation as the classifier's context window.  Where fewer
    than k prior same-ID frames exist inside the block, the earliest available
    one is repeated (edge padding).
    """
    n = len(df)
    hist = np.zeros((n, k), dtype=np.int32)
    ids = df["arbitration_id"].to_numpy()

    # positions grouped by (block, id), in time order
    key = pd.DataFrame({"b": block, "i": ids})
    for (_, _), pos in key.groupby(["b", "i"], sort=False).indices.items():
        pos = np.sort(pos)
        m = len(pos)
        for j in range(k):
            back = k - j
            src = np.maximum(np.arange(m) - back, 0)
            hist[pos, j] = pos[src]
    return hist


def build_ecu_streams(
    residuals: np.ndarray,      # [N, 8] per-frame Digital Twin residual
    gather_idx: np.ndarray,     # [N, WINDOW]
    id_index: np.ndarray,       # [N] dense stream index per frame
    n_streams: int,
    batch: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean residual per ECU stream inside each context window.

    Returns streams [N, n_streams, 8] and a presence mask [N, n_streams].
    Computed in batches so the intermediate [batch, WINDOW, 8] tensor stays small.
    """
    n, w = gather_idx.shape
    streams = np.zeros((n, n_streams, residuals.shape[1]), dtype=np.float32)
    counts = np.zeros((n, n_streams), dtype=np.float32)

    for s in range(0, n, batch):
        e = min(s + batch, n)
        g = gather_idx[s:e]                       # [b, W]
        b = e - s
        res_w = residuals[g]                      # [b, W, 8]
        sid = id_index[g]                         # [b, W]
        rows = np.repeat(np.arange(b), w)
        cols = sid.ravel()
        np.add.at(streams[s:e], (rows, cols), res_w.reshape(-1, residuals.shape[1]))
        np.add.at(counts[s:e], (rows, cols), 1.0)

    mask = (counts > 0).astype(np.float32)
    streams /= np.maximum(counts, 1.0)[..., None]
    return streams, mask


def rank_sort_streams(streams: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder ECU streams by residual magnitude so slot index carries no CAN ID.

    The fusion branch normally keeps one fixed slot per arbitration ID.  That is
    fine when the model is allowed to see CAN IDs, but in the ID-free controlled
    experiment it would leak the identity straight back in: "slot 1 is active"
    is equivalent to "ID 00000000 is on the bus", which alone identifies DoS.

    Sorting each window's streams by mean residual (descending, absent streams
    last) keeps the multi-ECU structure -- several independent residual streams,
    fused by attention -- while making the slot index a *rank* rather than an
    identity, so no arbitration ID can be recovered from position.
    """
    key = streams.mean(axis=2)
    key = np.where(mask > 0, key, -np.inf)
    order = np.argsort(-key, axis=1)
    rows = np.arange(streams.shape[0])[:, None]
    return streams[rows, order], mask[rows, order]
