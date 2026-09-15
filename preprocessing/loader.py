"""
GEM-CAN dataset loader.

Adapts automatically to the *actual* files shipped in GEM_CAN_Dataset.zip
(Zenodo record 17834776) rather than assuming a schema.

Real quirks handled here:
  * The two attack CSVs ship with a corrupted first-column header
    ("tiTestaTp" instead of "timestamp").  We normalise any first column
    that looks like a timestamp.
  * `label` is 'R' (regular) / 'T' (attack), not 0/1.
  * Only GEM_Attack_Scenario_Raw_Labeled.csv carries `Attack_Type`.
  * Trailing blank line in GEM_Normal_Driving_Raw.csv.
  * arbitration_id and data bytes are hexadecimal strings.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical class vocabulary -- derived from the dataset, not assumed.
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["Normal", "DoS", "Brake Tampering", "Steering Tampering"]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

DATA_COLS: List[str] = [f"data{i}" for i in range(8)]

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "GEM_CAN_Dataset")

FILES = {
    "normal": "GEM_Normal_Driving_Raw.csv",
    "attack": "GEM_Attack_Scenario_Raw_Labeled.csv",
    "attack_unlabeled": "GEM_Attack_Scenario_Raw.csv",
}


def dataset_dir(path: Optional[str] = None) -> str:
    return path or os.environ.get("GEMCAN_DIR", _DEFAULT_DIR)


def dataset_available(path: Optional[str] = None) -> bool:
    d = dataset_dir(path)
    return all(os.path.exists(os.path.join(d, FILES[k])) for k in ("normal", "attack"))


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and repair the known corrupted timestamp header."""
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    if first.lower() != "timestamp":
        # The published attack CSVs have 'tiTestaTp'. Any first column that is
        # not already 'timestamp' but parses as a float series is the timestamp.
        probe = pd.to_numeric(df[first], errors="coerce")
        if probe.notna().mean() > 0.9:
            df = df.rename(columns={first: "timestamp"})
    return df


def _hex_series(s: pd.Series, default: int = 0) -> pd.Series:
    """Vectorised hex -> int for CAN byte / id columns."""
    cleaned = (
        s.astype(str)
         .str.strip()
         .str.replace("^0[xX]", "", regex=True)
         .replace({"": None, "nan": None, "None": None})
    )
    out = cleaned.apply(lambda v: int(v, 16) if isinstance(v, str) else default)
    return out.astype(np.int64)


def load_raw(kind: str, path: Optional[str] = None) -> pd.DataFrame:
    """Load one GEM-CAN CSV and return a normalised frame-level DataFrame."""
    d = dataset_dir(path)
    fp = os.path.join(d, FILES[kind])
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"GEM-CAN file not found: {fp}\n"
            "Download GEM_CAN_Dataset.zip from https://zenodo.org/records/17834776 "
            "and extract it into the data/ directory."
        )

    df = pd.read_csv(fp, dtype=str, skip_blank_lines=True)
    df = _normalise_columns(df)

    # Drop rows with no arbitration id (trailing blank line in the normal file).
    df = df[df["arbitration_id"].notna()]
    df = df[df["arbitration_id"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    out["arbitration_id"] = _hex_series(df["arbitration_id"])
    out["arbitration_id_hex"] = (
        df["arbitration_id"].astype(str).str.strip().str.upper().str.replace("^0X", "", regex=True)
    )
    out["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(8).astype(np.int64)

    for c in DATA_COLS:
        out[c] = _hex_series(df[c]) if c in df.columns else 0

    # ---- class label -------------------------------------------------------
    if "Attack_Type" in df.columns:
        at = df["Attack_Type"].astype(str).str.strip()
        out["class_name"] = at.where(at.isin(CLASS_NAMES), "Normal")
    else:
        lab = df["label"].astype(str).str.strip().str.upper() if "label" in df.columns else "R"
        # No per-type annotation available -> binary fallback.
        out["class_name"] = np.where(pd.Series(lab, index=out.index) == "T", "DoS", "Normal")

    out["y"] = out["class_name"].map(CLASS_TO_IDX).astype(np.int64)
    out["is_attack"] = (out["y"] != CLASS_TO_IDX["Normal"]).astype(np.int64)
    out["source"] = kind

    out = out[out["timestamp"].notna()].reset_index(drop=True)
    # Frames are recorded in capture order; keep it (timestamps are monotonic
    # within a capture and define the replay ordering used by the demo).
    return out


def load_all(path: Optional[str] = None) -> pd.DataFrame:
    """Load normal-driving + labelled attack capture as one frame table.

    The two captures are separate recordings, so a `source` column is kept and
    all downstream splitting is done *within* a source, chronologically.
    """
    normal = load_raw("normal", path)
    attack = load_raw("attack", path)
    df = pd.concat([normal, attack], ignore_index=True)
    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["class_name"].value_counts()
    return (
        pd.DataFrame({"Class": counts.index, "Frames": counts.values})
        .assign(Share=lambda t: (t["Frames"] / t["Frames"].sum() * 100).round(2))
        .reset_index(drop=True)
    )


def present_classes(df: pd.DataFrame) -> List[str]:
    """Attack classes actually present -- never assume classes that aren't there."""
    return [c for c in CLASS_NAMES if c in set(df["class_name"].unique())]
