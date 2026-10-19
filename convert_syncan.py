# -*- coding: utf-8 -*-
"""
Convert the ETAS SynCAN dataset into the two CSVs this project reads, so
train.py and train_zeroday.py run on it unchanged.

    python convert_syncan.py --src <dir with SynCAN csv files> \
        --out data/SynCAN_as_GEM
    GEMCAN_DIR=data/SynCAN_as_GEM python train.py

SynCAN (Hanselmann et al., IEEE Access 2020, https://github.com/etas/SynCAN)
is a *signal-space* dataset: each row carries one to four already-normalised
signal values in [0, 1] rather than eight payload bytes. This project's twin
predicts payload bytes, so each signal is packed into the payload the way a
real DBC packs one: as a 16-bit big-endian unsigned integer, signal k into
bytes 2k and 2k+1. With at most four signals per ID that fits the eight-byte
payload exactly, and the quantisation step is 1/65535, far below the size of
any injected deviation in the dataset.

The label space is this project's four fixed slots, and SynCAN's classes are
not those. The assignment used here is semantic, not cosmetic:

    flooding    -> DoS                  high-rate injection on one ID
    plateau     -> Brake Tampering      a signal held at a constant value
    continuous  -> Steering Tampering   a signal drifting off its true value

playback and suppress have no slot and are therefore excluded from the
supervised runs. They are still usable through the label-free residual gate,
which is what evaluate_syncan_gate.py does with them.

Every number produced from this output describes a SynCAN attack, never a GEM
brake or steering attack. conversion.json records the assignment and the
converter prints it on every run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

NORMAL_FILE = "GEM_Normal_Driving_Raw.csv"
ATTACK_FILE = "GEM_Attack_Scenario_Raw_Labeled.csv"

# SynCAN test file -> this project's slot
SLOT_OF = {
    "test_flooding.csv": "DoS",
    "test_plateau.csv": "Brake Tampering",
    "test_continuous.csv": "Steering Tampering",
}

# Signal count per ID, from the dataset README: id1 .. id10
N_SIGNALS = {f"id{k}": n for k, n in
             enumerate((2, 3, 2, 1, 2, 2, 2, 1, 1, 4), start=1)}

# id1 -> 101, id10 -> 10A. Arbitrary but recorded, and distinct per ECU.
HEX_OF = {f"id{k}": format(0x100 + k, "03X") for k in range(1, 11)}


def read_syncan(path: str, nrows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [str(c).strip() for c in df.columns]
    sig = [c for c in df.columns if c.lower().startswith("signal")]
    if not sig:
        raise SystemExit(f"{os.path.basename(path)}: no Signal* columns, found "
                         f"{list(df.columns)}")
    keep = ["Label", "Time", "ID"] + sig
    missing = [c for c in ("Label", "Time", "ID") if c not in df.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(path)}: missing column(s) {missing}")
    return df[keep].rename(columns={c: f"s{i}" for i, c in enumerate(sig)})


def to_frames(df: pd.DataFrame, slot: str, bits: int = 16) -> pd.DataFrame:
    """Pack the signal columns into eight payload bytes, GEM-CAN schema."""
    ids = df["ID"].astype(str).str.strip()
    unknown = sorted(set(ids) - set(HEX_OF))
    if unknown:
        raise SystemExit(f"unexpected ID value(s): {unknown[:5]}")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df["Time"], errors="coerce") / 1000.0  # ms -> s
    out["arbitration_id"] = ids.map(HEX_OF)
    out["dlc"] = ids.map(N_SIGNALS).astype(int) * (2 if bits == 16 else 1)

    n_sig = sum(1 for c in df.columns if c.startswith("s"))
    for k in range(4):
        col = f"s{k}"
        if k < n_sig:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        else:
            v = np.zeros(len(df))
        v = np.nan_to_num(np.clip(v, 0.0, 1.0), nan=0.0)
        if bits == 16:
            q = np.rint(v * 65535.0).astype(np.int64)
            out[f"data{2 * k}"] = [format(x, "02X") for x in (q >> 8) & 0xFF]
            out[f"data{2 * k + 1}"] = [format(x, "02X") for x in q & 0xFF]
        else:
            q = np.rint(v * 255.0).astype(np.int64)
            out[f"data{k}"] = [format(x, "02X") for x in q & 0xFF]
    if bits == 8:
        for k in range(4, 8):
            out[f"data{k}"] = "00"

    lab = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    out["Attack_Type"] = np.where(lab == 1, slot, "Normal")
    return out[out["timestamp"].notna()].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory holding the SynCAN CSVs")
    ap.add_argument("--out", required=True, help="directory to write the two CSVs into")
    ap.add_argument("--normal-rows", type=int, default=100_000,
                    help="rows taken from train_1.csv for the attack-free capture")
    ap.add_argument("--attack-rows", type=int, default=50_000,
                    help="rows taken from each attack test file")
    ap.add_argument("--bits", type=int, default=16, choices=[8, 16],
                    help="bits per signal in the packed payload. 16 is the "
                         "faithful encoding; 8 exists as a control, to show "
                         "that the gate's behaviour here is not an artefact "
                         "of the quantisation")
    args = ap.parse_args()

    print("=" * 74)
    print("CLASS SLOT ASSIGNMENT -- the output uses this project's slot names,")
    print("but every one of them stands for a SynCAN attack:")
    for f, s in SLOT_OF.items():
        print(f"    {f[5:-4]:<12} occupies slot  {s:<20} <-- RELABEL every figure")
    print("    playback, suppress  have no slot; gate-only evaluation instead")
    print("=" * 74)

    os.makedirs(args.out, exist_ok=True)

    # ----------------------------------------------------------------- normal
    src_norm = os.path.join(args.src, "train_1.csv")
    if not os.path.exists(src_norm):
        raise SystemExit(f"not found: {src_norm}")
    nf = to_frames(read_syncan(src_norm, args.normal_rows), "Normal", args.bits)
    if (nf["Attack_Type"] != "Normal").any():
        raise SystemExit("train_1.csv carried an attack label; refusing to use it "
                         "as the attack-free capture")
    nf.to_csv(os.path.join(args.out, NORMAL_FILE), index=False)
    print(f"\n  normal capture: {len(nf):,} frames -> {NORMAL_FILE}")

    # ----------------------------------------------------------------- attack
    # Each test file has its own time base. Lay them end to end rather than
    # interleaving them, so every context window stays inside one capture.
    parts: List[pd.DataFrame] = []
    cursor = float(nf["timestamp"].max()) + 1.0
    for fname, slot in SLOT_OF.items():
        p = os.path.join(args.src, fname)
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")
        af = to_frames(read_syncan(p, args.attack_rows), slot, args.bits)
        af["timestamp"] = af["timestamp"] - af["timestamp"].min() + cursor
        cursor = float(af["timestamp"].max()) + 1.0
        parts.append(af)
        n_atk = int((af["Attack_Type"] == slot).sum())
        print(f"  {fname:<22} {len(af):>8,} frames, {n_atk:>7,} attack "
              f"({n_atk / len(af) * 100:5.2f}%) -> slot {slot}")

    atk = pd.concat(parts, ignore_index=True)
    atk.to_csv(os.path.join(args.out, ATTACK_FILE), index=False)
    print(f"\n  attack capture: {len(atk):,} frames -> {ATTACK_FILE}")
    print("  class counts:", atk["Attack_Type"].value_counts().to_dict())

    with open(os.path.join(args.out, "conversion.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "SynCAN (ETAS), https://github.com/etas/SynCAN",
                   "citation": "Hanselmann et al., CANet, IEEE Access 8:58194, 2020",
                   "encoding": (f"each signal as uint{args.bits}"
                                + (" big-endian, signal k -> payload bytes 2k, 2k+1"
                                   if args.bits == 16 else ", signal k -> payload byte k")),
                   "bits_per_signal": args.bits,
                   "slot_assignment": {f[5:-4]: s for f, s in SLOT_OF.items()},
                   "excluded": ["playback", "suppress"],
                   "id_map": HEX_OF,
                   "normal_rows": args.normal_rows,
                   "attack_rows_per_file": args.attack_rows,
                   "normal_frames": int(len(nf)),
                   "attack_frames": int(len(atk))}, fh, indent=2)

    # --------------------------------------------------- verify it round-trips
    print("\n  Verifying with this project's own loader ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing.loader import load_all
    df = load_all(args.out)
    counts = {k: int(v) for k, v in df["class_name"].value_counts().items()}
    print("   ", counts)
    if set(counts) == {"Normal"}:
        raise SystemExit("every frame loaded as Normal -- the labels did not survive.")
    for slot in SLOT_OF.values():
        if counts.get(slot, 0) == 0:
            raise SystemExit(f"slot {slot!r} is empty after loading.")
    print("\n  Ready:")
    print(f"    GEMCAN_DIR={args.out} python train.py")
    print("\n  Remember to relabel results using conversion.json.")


if __name__ == "__main__":
    main()
