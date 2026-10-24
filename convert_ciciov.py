# -*- coding: utf-8 -*-
"""
Convert the CICIoV2024 dataset into the two CSVs this project reads, so
train.py and train_zeroday.py run on it unchanged.

    python convert_ciciov.py --src <dir with the decimal_*.csv files> \
        --out data/CICIoV2024_as_GEM
    GEMCAN_DIR=data/CICIoV2024_as_GEM python train.py

CICIoV2024 (Canadian Institute for Cybersecurity, 2024) is a real capture from
a 2019 Ford vehicle. Two of its properties shape this converter.

First, it ships no timestamp and no data length code: the schema is
ID, DATA_0..DATA_7, label, category, specific_class. A timestamp is therefore
synthesised from row order at a nominal 1 ms interval and the data length code
is set to 8 for every frame. The consequence has to be stated wherever results
from this dataset are reported: the inter-arrival feature is constant, so it
carries no information here, and no timing-based claim can be made on it.

Second, the label space of this project has three attack slots and CICIoV2024
has five attack classes. The assignment is semantic and recorded in
conversion.json:

    DoS              -> DoS                  denial of service on one identifier
    STEERING_WHEEL   -> Steering Tampering   steering signal spoofing
    GAS, RPM, SPEED  -> Brake Tampering      powertrain signal spoofing, grouped

The third slot therefore stands for grouped powertrain spoofing, not for brake
tampering, and every figure drawn from this output must say so. The converter
prints the mapping on every run.
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

SLOT_OF = {
    "decimal_DoS.csv": ("DoS", "DoS"),
    "decimal_spoofing-STEERING_WHEEL.csv": ("Steering Tampering", "steering spoofing"),
    "decimal_spoofing-GAS.csv": ("Brake Tampering", "powertrain spoofing (GAS)"),
    "decimal_spoofing-RPM.csv": ("Brake Tampering", "powertrain spoofing (RPM)"),
    "decimal_spoofing-SPEED.csv": ("Brake Tampering", "powertrain spoofing (SPEED)"),
}
DATA = [f"DATA_{i}" for i in range(8)]
DT = 0.001          # nominal inter-frame interval, seconds


def read_src(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in ["ID"] + DATA if c not in df.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(path)}: missing column(s) {missing}")
    return df


def to_frames(df: pd.DataFrame, slot: str, t0: float) -> pd.DataFrame:
    # A slice taken with .iloc keeps the source index. Assigning a Series from it
    # into a fresh frame aligns on that index and silently yields all-NaN, so the
    # index is dropped here rather than relied on.
    df = df.reset_index(drop=True)
    out = pd.DataFrame()
    out["timestamp"] = t0 + np.arange(len(df), dtype=float) * DT
    out["arbitration_id"] = [format(int(v), "03X") for v in df["ID"]]
    out["dlc"] = 8
    for k, c in enumerate(DATA):
        v = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 255)
        out[f"data{k}"] = [format(x, "02X") for x in v]
    out["Attack_Type"] = slot
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--normal-rows", type=int, default=120_000,
                    help="benign rows used as the attack-free capture")
    ap.add_argument("--context-rows", type=int, default=20_000,
                    help="benign rows placed before each attack segment, so a "
                         "context window at the start of a segment has normal history")
    ap.add_argument("--cap", type=int, default=30_000,
                    help="maximum attack rows taken per slot")
    args = ap.parse_args()

    print("=" * 74)
    print("CLASS SLOT ASSIGNMENT -- the output uses this project's slot names,")
    print("but every one of them stands for a CICIoV2024 class:")
    for f, (slot, what) in SLOT_OF.items():
        print(f"    {what:<28} occupies slot  {slot:<20} <-- RELABEL every figure")
    print("=" * 74)

    os.makedirs(args.out, exist_ok=True)
    benign_path = os.path.join(args.src, "decimal_benign.csv")
    if not os.path.exists(benign_path):
        raise SystemExit(f"not found: {benign_path}")

    need = args.normal_rows + args.context_rows * 3
    benign = read_src(benign_path, need)
    if len(benign) < need:
        raise SystemExit(f"benign capture has {len(benign):,} rows, need {need:,}")

    # ------------------------------------------------------------- normal
    nf = to_frames(benign.iloc[:args.normal_rows], "Normal", 0.0)
    nf.to_csv(os.path.join(args.out, NORMAL_FILE), index=False)
    print(f"\n  normal capture: {len(nf):,} frames -> {NORMAL_FILE}")

    # ------------------------------------------------------------- attack
    by_slot: Dict[str, List[pd.DataFrame]] = {}
    for fname, (slot, what) in SLOT_OF.items():
        p = os.path.join(args.src, fname)
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")
        by_slot.setdefault(slot, []).append(read_src(p))

    parts: List[pd.DataFrame] = []
    cursor = float(nf["timestamp"].max()) + 1.0
    ctx_at = args.normal_rows
    for slot, frames in by_slot.items():
        rows = pd.concat(frames, ignore_index=True)
        if len(rows) > args.cap:
            rows = rows.iloc[:args.cap]
        ctx = benign.iloc[ctx_at:ctx_at + args.context_rows]
        ctx_at += args.context_rows
        seg = pd.concat([to_frames(ctx, "Normal", cursor),
                         to_frames(rows, slot, cursor + len(ctx) * DT)],
                        ignore_index=True)
        cursor = float(seg["timestamp"].max()) + 1.0
        parts.append(seg)
        print(f"  {slot:<20} {len(rows):>8,} attack frames "
              f"(+{len(ctx):,} normal context)")

    atk = pd.concat(parts, ignore_index=True)
    atk.to_csv(os.path.join(args.out, ATTACK_FILE), index=False)
    print(f"\n  attack capture: {len(atk):,} frames -> {ATTACK_FILE}")
    print("  class counts:", atk["Attack_Type"].value_counts().to_dict())

    with open(os.path.join(args.out, "conversion.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "CICIoV2024, Canadian Institute for Cybersecurity",
                   "vehicle": "2019 Ford",
                   "synthesised": "timestamp from row order at a nominal 1 ms "
                                  "interval; dlc fixed at 8. The inter-arrival "
                                  "feature therefore carries no information.",
                   "slot_assignment": {w: s for _, (s, w) in SLOT_OF.items()},
                   "normal_frames": int(len(nf)),
                   "attack_frames": int(len(atk))}, fh, indent=2)

    print("\n  Verifying with this project's own loader ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing.loader import load_all
    df = load_all(args.out)
    counts = {k: int(v) for k, v in df["class_name"].value_counts().items()}
    print("   ", counts)
    for slot in ("DoS", "Brake Tampering", "Steering Tampering"):
        if counts.get(slot, 0) == 0:
            raise SystemExit(f"slot {slot!r} is empty after loading.")
    print("\n  Ready:")
    print(f"    GEMCAN_DIR={args.out} python train.py")
    print("\n  Remember to relabel results using conversion.json.")


if __name__ == "__main__":
    main()
