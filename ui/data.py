"""
Cached data + model access for the Streamlit app.

Everything the UI displays originates here, and everything here originates from
either the real GEM-CAN CSVs or a trained checkpoint.  Nothing is synthesised.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from evaluation.audit import run_full_audit
from models.digital_twin import GRUDigitalTwin
from models.residual_fusion import TwinShieldFusion
from preprocessing.loader import CLASS_NAMES, DATA_COLS, dataset_available
from preprocessing.prepare import Prepared, prepare
from preprocessing.windowing import build_ecu_streams, build_id_history

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED = os.path.join(ROOT, "saved_models")
RESULTS = os.path.join(ROOT, "results")

DEFAULT_VARIANT = "with_id"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def data_ready() -> bool:
    return dataset_available()


def models_ready(variant: str = DEFAULT_VARIANT) -> bool:
    return (os.path.exists(os.path.join(SAVED, "digital_twin.pt"))
            and os.path.exists(os.path.join(SAVED, f"{variant}__twinshield__fusion.pt"))
            and os.path.exists(os.path.join(RESULTS, "results.json")))


def missing_notice() -> None:
    st.markdown(
        '<div class="ts-flag">⚠️ <b>Artifacts not found.</b> Run '
        '<code>python train.py</code> once before the demo. '
        'The pages below need <code>saved_models/</code> and '
        '<code>results/results.json</code>.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Core data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GEM-CAN dataset ...")
def get_prepared() -> Prepared:
    return prepare()


@st.cache_resource(show_spinner=False)
def get_id_index() -> np.ndarray:
    p = get_prepared()
    return np.ascontiguousarray(
        p.df["arbitration_id"].map(lambda v: p.id_vocab.get(int(v), 0)).to_numpy(),
        dtype=np.int64)


@st.cache_resource(show_spinner="Loading Digital Twin outputs ...")
def get_twin_outputs() -> Tuple[np.ndarray, np.ndarray]:
    """(residual, expected_payload) -- real GRU outputs saved during training."""
    res = np.load(os.path.join(SAVED, "residuals.npy"))
    exp = np.load(os.path.join(SAVED, "expected_payload.npy"))
    return res, exp


@st.cache_resource(show_spinner="Building ECU residual streams ...")
def get_streams() -> Tuple[np.ndarray, np.ndarray]:
    """Per-ECU residual streams, cached to disk so the demo never rebuilds them."""
    cache = os.path.join(SAVED, "ecu_streams.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["streams"], z["mask"]

    p = get_prepared()
    residual, _ = get_twin_outputs()
    id_index = get_id_index()
    streams, mask = build_ecu_streams(residual, p.windows.gather_idx, id_index,
                                      len(p.id_vocab) + 1)
    try:
        np.savez(cache, streams=streams, mask=mask)
    except OSError:
        pass  # read-only checkout is fine; we just recompute next time
    return streams, mask


@st.cache_resource(show_spinner=False)
def get_detector_features() -> np.ndarray:
    """Feature matrix for the deployed TwinShield detector (raw + residual)."""
    p = get_prepared()
    residual, _ = get_twin_outputs()
    return np.concatenate([p.features(True), residual], axis=1).astype(np.float32)


@st.cache_resource(show_spinner="Loading detector ...")
def get_detector(variant: str = DEFAULT_VARIANT) -> TwinShieldFusion:
    p = get_prepared()
    X = get_detector_features()
    model = TwinShieldFusion(X.shape[1], len(CLASS_NAMES), len(p.id_vocab) + 1)
    state = torch.load(os.path.join(SAVED, f"{variant}__twinshield__fusion.pt"),
                       map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def get_digital_twin() -> GRUDigitalTwin:
    p = get_prepared()
    n_ids = len(p.id_vocab) + 1
    model = GRUDigitalTwin(n_ids=n_ids)
    model.load_state_dict(torch.load(os.path.join(SAVED, "digital_twin.pt"),
                                     map_location="cpu"))
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def get_results() -> Dict:
    with open(os.path.join(RESULTS, "results.json")) as fh:
        return json.load(fh)


@st.cache_data(show_spinner="Auditing dataset ...")
def get_audit() -> Dict:
    a = run_full_audit(get_prepared())
    return {
        "class_balance": a["class_balance"].to_dict("records"),
        "duplicates": a["duplicates"],
        "can_id_overlap": a["can_id_overlap"].to_dict("records"),
        "can_id_shortcut": a["can_id_shortcut"],
    }


# ---------------------------------------------------------------------------
# Live inference -- genuinely runs the model
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer(frame_idx: np.ndarray, variant: str = DEFAULT_VARIANT
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Run the trained TwinShield detector on the given frames.

    Returns (probabilities [n, n_classes], attention [n, n_streams]).
    This is a real forward pass -- the Live Monitor calls it while replaying.
    """
    p = get_prepared()
    X = get_detector_features()
    streams, mask = get_streams()
    model = get_detector(variant)

    g = p.windows.gather_idx[frame_idx]
    x = torch.from_numpy(X[g])
    st_t = torch.from_numpy(streams[frame_idx])
    mk_t = torch.from_numpy(mask[frame_idx])
    logits, alpha = model.forward_with_attention(x, st_t, mk_t)
    return torch.softmax(logits, 1).numpy(), alpha.numpy()


# ---------------------------------------------------------------------------
# Replay scenarios -- contiguous slices of the real capture
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_scenarios(lead: int = 70, length: int = 190) -> Dict[str, Dict]:
    """Replay scenarios built from real frames only.

    An important property of GEM-CAN: the attack capture is *already under
    attack from its very first frame* -- brake tampering begins at t = 8.2 s and
    repeats throughout.  There is therefore no stretch inside that recording
    that runs clean and then turns malicious.

    So each scenario is assembled from two real recordings played back to back:
    a lead-in taken from the normal-driving capture, followed by a slice of the
    attack capture.  Nothing is generated and no frame is altered -- every frame
    is still classified using its own true preceding context, because context
    comes from each frame's own block.  The UI states plainly that this is a
    join of two captures rather than one continuous drive.
    """
    p = get_prepared()
    df = p.df
    y = p.windows.y
    split = p.windows.split
    src = df["source"].to_numpy()

    normal_idx = np.where(src == "normal")[0]
    attack_idx = np.where(src == "attack")[0]
    if len(normal_idx) == 0 or len(attack_idx) == 0:
        return {}
    atk0, atk1 = int(attack_idx.min()), int(attack_idx.max())

    # Lead-in: a run of genuinely normal frames taken from a held-out block, so
    # the quiet part of the demo is also data the model never trained on.
    lead_start = None
    blocks = p.windows.block_id
    for b in np.unique(blocks[normal_idx]):
        pos = np.where((blocks == b) & (src == "normal"))[0]
        if len(pos) < lead:
            continue
        seg = slice(int(pos.min()), int(pos.min()) + lead)
        if (y[seg] == 0).all() and (split[seg] == "test").all():
            lead_start = seg.start
            break
    if lead_start is None:                       # no fully held-out run available
        lead_start = int(normal_idx.max()) - lead
    lead_range = (lead_start, lead_start + lead)

    def describe(idx: np.ndarray) -> Dict[str, int]:
        return {CLASS_NAMES[c]: int((y[idx] == c).sum())
                for c in range(len(CLASS_NAMES)) if (y[idx] == c).sum()}

    def pick_attack_segment(classes: Tuple[int, ...]) -> int:
        """Best `length`-frame window carrying these classes, preferring held-out data."""
        best, best_score = atk0, -1e9
        for start in range(atk0, atk1 - length, 16):
            seg = slice(start, start + length)
            n_cls = int(np.isin(y[seg], classes).sum())
            if n_cls < 20:
                continue
            test_share = float((split[seg] == "test").mean())
            score = test_share * 3.0 + min(n_cls, 80) / 80.0
            if score > best_score:
                best, best_score = start, score
        return best

    def build(name: str, a_start: int) -> Optional[Tuple[str, Dict]]:
        a_start = max(atk0, min(a_start, atk1 - length))
        a_idx = np.arange(a_start, min(a_start + length, atk1 + 1))
        l_idx = np.arange(*lead_range)
        idx = np.concatenate([l_idx, a_idx])
        return name, {
            "frames": idx,
            "lead": len(l_idx),
            "attack_range": (int(a_idx.min()), int(a_idx.max())),
            "lead_range": lead_range,
            "test_share": round(float((split[idx] == "test").mean()) * 100, 1),
            "composition": describe(idx),
        }

    out: Dict[str, Dict] = {}

    if (y == 1).any():
        name, sc = build("DoS Flood (ID 00000000)", pick_attack_segment((1,)))
        out[name] = sc

    # Brake and steering tampering are injected as an interleaved pair 0.56 s
    # apart, so they occupy the same stretch of bus traffic and are presented as
    # one scenario rather than two near-identical menu entries.
    if ((y == 2) | (y == 3)).any():
        name, sc = build("Brake + Steering Tampering", pick_attack_segment((2, 3)))
        out[name] = sc

    return out


def frame_table(idx: np.ndarray, preds: np.ndarray, confs: np.ndarray) -> pd.DataFrame:
    """Human-readable CAN frames, straight from the dataset."""
    p = get_prepared()
    df = p.df.iloc[idx]
    payload = df[DATA_COLS].to_numpy(dtype=int)
    return pd.DataFrame({
        "Time": df["timestamp"].to_numpy(),
        "CAN ID": df["arbitration_id_hex"].to_numpy(),
        "DLC": df["dlc"].to_numpy(),
        "Payload": [" ".join(f"{b:02X}" for b in row) for row in payload],
        "Predicted": [CLASS_NAMES[c] for c in preds],
        "Confidence": confs,
        "Actual": df["class_name"].to_numpy(),
    })


def id_stream_labels() -> List[str]:
    p = get_prepared()
    inv = {v: k for k, v in p.id_vocab.items()}
    return ["<unknown>"] + [f"{inv[i]:08X}" for i in range(1, len(p.id_vocab) + 1)]
