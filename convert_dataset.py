# -*- coding: utf-8 -*-
"""
Convert a third-party CAN intrusion dataset into the two CSVs this project
reads, so train.py and train_zeroday.py run on it unchanged.

    python convert_dataset.py --preset car-hacking \
        --src /path/to/Car-Hacking --out data/CarHacking_as_GEM
    GEMCAN_DIR=data/CarHacking_as_GEM python train.py

A caution that matters more than the mechanics. This project's label space is
fixed at four slots:

    Normal, DoS, Brake Tampering, Steering Tampering

Another dataset has its own classes, and they are not those. The converter
therefore places each foreign attack class into a slot and records the
assignment in conversion.json. Slot names in the output are the project's;
the classes they stand for are the source dataset's. Every figure and table
produced from converted data has to be relabelled using that file, and saying
"brake tampering" about a Car-Hacking RPM-spoofing run would be a false
statement about an experiment. The converter prints the mapping on every run
so it cannot be forgotten.

Presets:
    car-hacking   HCRL Car-Hacking: Timestamp, CAN ID, DLC, DATA[0..7], Flag,
                  one file per attack type plus a normal run
    generic       any CSV, with the column names given on the command line

Generic use:
    python convert_dataset.py --preset generic \
        --normal normal.csv --attack dos.csv=DoS --attack fuzzy.csv=Fuzzy \
        --col-time Timestamp --col-id "CAN ID" --col-dlc DLC \
        --col-data "DATA[0]" --flag-col Flag --flag-attack T \
        --out data/Other_as_GEM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SLOTS = ["DoS", "Brake Tampering", "Steering Tampering"]   # Normal is slot 0
NORMAL_FILE = "GEM_Normal_Driving_Raw.csv"
ATTACK_FILE = "GEM_Attack_Scenario_Raw_Labeled.csv"

CAR_HACKING = {
    "DoS_dataset.csv": "DoS",
    "Fuzzy_dataset.csv": "Fuzzy",
    "gear_dataset.csv": "Gear spoofing",
    "RPM_dataset.csv": "RPM spoofing",
}


def read_any(path: str, cols: Dict[str, str], data0: str) -> pd.DataFrame:
    """Read one source file into the eight-byte frame table this project wants."""
    df = pd.read_csv(path, header=0, dtype=str, skip_blank_lines=True,
                     on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [v for k, v in cols.items() if v and v not in df.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(path)}: column(s) not found: {missing}\n"
                         f"  available: {list(df.columns)[:14]}")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df[cols["time"]], errors="coerce")
    out["arbitration_id"] = (df[cols["id"]].astype(str).str.strip()
                             .str.replace("^0[xX]", "", regex=True).str.upper())
    out["dlc"] = pd.to_numeric(df[cols["dlc"]], errors="coerce").fillna(8).astype(int)

    # payload columns: either DATA[0..7] style, or eight consecutive columns
    names = list(df.columns)
    if data0 in names:
        start = names.index(data0)
        pay = names[start:start + 8]
    else:
        raise SystemExit(f"payload column {data0!r} not found in {os.path.basename(path)}")
    for k, c in enumerate(pay):
        out[f"data{k}"] = df[c]
    return out, df


def to_long(out: pd.DataFrame) -> pd.DataFrame:
    """Normalise payload bytes to decimal integers 0-255."""
    for k in range(8):
        col = out[f"data{k}"].astype(str).str.strip()
        val = col.apply(lambda v: int(v, 16) if v and v.lower() not in
                        ("nan", "none", "") and all(ch in "0123456789abcdefABCDEF"
                                                    for ch in v) else 0)
        out[f"data{k}"] = np.clip(val.to_numpy(), 0, 255)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["car-hacking", "generic"], required=True)
    ap.add_argument("--src", help="directory holding the source dataset")
    ap.add_argument("--out", required=True, help="directory to write the two CSVs into")
    ap.add_argument("--normal", help="generic: the attack-free file")
    ap.add_argument("--attack", action="append", default=[],
                    help="generic: file.csv=ClassName, repeatable")
    ap.add_argument("--col-time", default="Timestamp")
    ap.add_argument("--col-id", default="CAN ID")
    ap.add_argument("--col-dlc", default="DLC")
    ap.add_argument("--col-data", default="DATA[0]")
    ap.add_argument("--flag-col", default="Flag")
    ap.add_argument("--flag-attack", default="T")
    ap.add_argument("--slot", action="append", default=[],
                    help="force an assignment, e.g. --slot \"Fuzzy=Brake Tampering\"")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep at most this many frames per source file")
    args = ap.parse_args()

    cols = {"time": args.col_time, "id": args.col_id, "dlc": args.col_dlc}

    # ------------------------------------------------- gather the source files
    if args.preset == "car-hacking":
        if not args.src:
            raise SystemExit("--src is required for the car-hacking preset")
        normal_path = None
        for cand in ("normal_run_data.csv", "normal_run_data.txt"):
            if os.path.exists(os.path.join(args.src, cand)):
                normal_path = os.path.join(args.src, cand)
        attacks = [(os.path.join(args.src, f), c) for f, c in CAR_HACKING.items()
                   if os.path.exists(os.path.join(args.src, f))]
        if not attacks:
            raise SystemExit(f"no Car-Hacking attack files found in {args.src}")
    else:
        normal_path = args.normal
        attacks = []
        for spec in args.attack:
            if "=" not in spec:
                raise SystemExit(f"--attack expects file.csv=ClassName, got {spec!r}")
            f, c = spec.split("=", 1)
            attacks.append((f, c))
        if not attacks:
            raise SystemExit("at least one --attack file.csv=ClassName is required")

    # ------------------------------------------------------- slot assignment
    forced = dict(s.split("=", 1) for s in args.slot if "=" in s)
    source_classes = [c for _, c in attacks]
    mapping: Dict[str, str] = {}
    free = [s for s in SLOTS]
    for c in source_classes:
        if c in forced:
            mapping[c] = forced[c]
            if forced[c] in free:
                free.remove(forced[c])
    for c in source_classes:
        if c in mapping:
            continue
        if c in SLOTS:                       # a class whose name already matches
            mapping[c] = c
            if c in free:
                free.remove(c)
    for c in source_classes:
        if c in mapping:
            continue
        if not free:
            raise SystemExit(
                f"more attack classes ({len(source_classes)}) than the three slots "
                f"this project supports. Choose which to keep with repeated "
                f"--attack, or assign explicitly with --slot.")
        mapping[c] = free.pop(0)

    print("=" * 74)
    print("CLASS SLOT ASSIGNMENT -- the output uses this project's slot names,")
    print("but they stand for the source dataset's classes:")
    for c, s in mapping.items():
        note = "  (name matches)" if c == s else "  <-- RELABEL every figure"
        print(f"    {c:<24} occupies slot  {s:<22}{note}")
    print("=" * 74)

    # ----------------------------------------------------------------- normal
    os.makedirs(args.out, exist_ok=True)
    if normal_path and os.path.exists(normal_path):
        nf, _ = read_any(normal_path, cols, args.col_data)
        nf = to_long(nf)
        if args.limit:
            nf = nf.head(args.limit)
        nf["Attack_Type"] = "Normal"
        nf = nf.dropna(subset=["timestamp"]).reset_index(drop=True)
        nf.to_csv(os.path.join(args.out, NORMAL_FILE), index=False)
        print(f"\n  normal capture: {len(nf):,} frames -> {NORMAL_FILE}")
    else:
        raise SystemExit("an attack-free capture is required; give it with --normal")

    # ----------------------------------------------------------------- attack
    parts: List[pd.DataFrame] = []
    for path, cls in attacks:
        af, raw = read_any(path, cols, args.col_data)
        af = to_long(af)
        if args.flag_col and args.flag_col in raw.columns:
            is_atk = (raw[args.flag_col].astype(str).str.strip().str.upper()
                      == args.flag_attack.upper()).to_numpy()
        else:
            is_atk = np.ones(len(af), dtype=bool)
            print(f"    [warn] {os.path.basename(path)}: no {args.flag_col!r} column, "
                  f"treating every row as attack")
        af["Attack_Type"] = np.where(is_atk, mapping[cls], "Normal")
        if args.limit:
            af = af.head(args.limit)
        af = af.dropna(subset=["timestamp"]).reset_index(drop=True)
        parts.append(af)
        print(f"  {os.path.basename(path):<28} {len(af):>9,} frames, "
              f"{int(is_atk[:len(af)].sum()):>8,} attack -> slot {mapping[cls]}")

    atk = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    atk = atk.reset_index(drop=True)
    atk.to_csv(os.path.join(args.out, ATTACK_FILE), index=False)
    print(f"\n  attack capture: {len(atk):,} frames -> {ATTACK_FILE}")
    print("  class counts:", atk["Attack_Type"].value_counts().to_dict())

    with open(os.path.join(args.out, "conversion.json"), "w", encoding="utf-8") as fh:
        json.dump({"preset": args.preset, "source": args.src or "",
                   "slot_assignment": mapping,
                   "normal_frames": int(len(nf)), "attack_frames": int(len(atk))},
                  fh, indent=2)

    # --------------------------------------------------- verify it round-trips
    print("\n  Verifying with this project's own loader ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing.loader import load_all
    df = load_all(args.out)
    counts = df["class_name"].value_counts().to_dict()
    print("   ", {k: int(v) for k, v in counts.items()})
    if set(counts) == {"Normal"}:
        raise SystemExit("every frame loaded as Normal -- the labels did not survive. "
                         "Check --flag-col and --flag-attack.")
    print("\n  Ready. Point the pipeline at it:")
    print(f"    GEMCAN_DIR={args.out} python train.py")
    print(f"    GEMCAN_DIR={args.out} python train_zeroday.py --unknown \"{list(mapping.values())[0]}\"")
    print("\n  Remember to relabel results using conversion.json.")


if __name__ == "__main__":
    main()
