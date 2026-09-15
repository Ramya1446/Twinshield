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
from models.anomaly_gate import ResidualAnomalyGate
from models.bilstm import BiLSTMDetector
from models.cnn import CNNDetector
from models.cnn_bilstm import CNNBiLSTMDetector
from models.digital_twin import GRUDigitalTwin
from models.residual_fusion import TwinShieldFusion
from preprocessing.loader import CLASS_NAMES, CLASS_TO_IDX, DATA_COLS, dataset_available
from preprocessing.prepare import Prepared, prepare
from preprocessing.windowing import build_ecu_streams

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
# Zero-day artifacts -- everything refitted with one attack class quarantined
# ---------------------------------------------------------------------------
ZD_ARCHS = {"CNN": "cnn", "BiLSTM": "bilstm", "CNN-BiLSTM": "cnn_bilstm",
            "TwinShield-Fusion": "twinshield"}
ZD_FILE = os.path.join(RESULTS, "zero_day.json")


def _zd(name: str) -> str:
    return os.path.join(SAVED, f"zeroday__{name}")


def zeroday_ready() -> bool:
    needed = [ZD_FILE, _zd("digital_twin.pt"), _zd("residuals.npy"),
              _zd("ecu_streams.npz"), _zd("gate.npz")]
    needed += [_zd(f"{a}.pt") for a in ZD_ARCHS.values()]
    return all(os.path.exists(f) for f in needed)


def zeroday_notice() -> None:
    st.markdown(
        '<div class="ts-flag">⚠️ <b>Zero-day experiment not run yet.</b> Run '
        '<code>python train_zeroday.py</code> once before the demo. It holds one '
        'attack class out of every training stage and refits the Digital Twin, '
        'the anomaly gate and all four detectors without it.</div>',
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_zero_day() -> Dict:
    with open(ZD_FILE) as fh:
        return json.load(fh)


@st.cache_resource(show_spinner="Loading zero-day Digital Twin outputs ...")
def get_zd_twin_outputs() -> Tuple[np.ndarray, np.ndarray]:
    """(residual, expected_payload) from the twin refitted without the unknown class."""
    return np.load(_zd("residuals.npy")), np.load(_zd("expected_payload.npy"))


@st.cache_resource(show_spinner=False)
def get_zd_streams() -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(_zd("ecu_streams.npz"))
    return z["streams"], z["mask"]


@st.cache_resource(show_spinner=False)
def get_gate() -> ResidualAnomalyGate:
    return ResidualAnomalyGate.load(_zd("gate.npz"))


@st.cache_resource(show_spinner=False)
def get_zd_features() -> Tuple[np.ndarray, np.ndarray]:
    """(raw features, raw + residual features) for the zero-day detectors."""
    p = get_prepared()
    zd = get_zero_day()
    residual, _ = get_zd_twin_outputs()
    base = p.features(bool(zd.get("use_id", True)))
    return base, np.concatenate([base, residual], axis=1).astype(np.float32)


@st.cache_resource(show_spinner="Loading the four zero-day detectors ...")
def get_zd_models() -> Dict[str, torch.nn.Module]:
    """All four detectors, each trained on a label space without the unknown class."""
    p = get_prepared()
    zd = get_zero_day()
    n_known = len(zd["known_classes"])
    n_streams = len(p.id_vocab) + 1
    X_raw, X_fus = get_zd_features()

    out: Dict[str, torch.nn.Module] = {}
    for name, key in ZD_ARCHS.items():
        if key == "cnn":
            m = CNNDetector(X_raw.shape[1], n_known)
        elif key == "bilstm":
            m = BiLSTMDetector(X_raw.shape[1], n_known)
        elif key == "cnn_bilstm":
            m = CNNBiLSTMDetector(X_raw.shape[1], n_known)
        else:
            m = TwinShieldFusion(X_fus.shape[1], n_known, n_streams)
        m.load_state_dict(torch.load(_zd(f"{key}.pt"), map_location="cpu"))
        m.eval()
        out[name] = m
    return out


@torch.no_grad()
def infer_all(frame_idx: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """Run all four zero-day detectors plus the anomaly gate on the given frames.

    Every value returned is produced by a real forward pass on this call; the
    Live Monitor invokes it once per replay tick.  Returns, per model:

        proba   [n, n_known]   softmax over the classes that model was trained on
        pred    [n]            argmax
        alarm   [n]            did this model raise an alert at all
        gate    [n] bool       gate contribution (TwinShield-Fusion only)

    plus a shared "gate" entry carrying the anomaly score and its explanation.
    """
    p = get_prepared()
    X_raw, X_fus = get_zd_features()
    streams, mask = get_zd_streams()
    residual, _ = get_zd_twin_outputs()
    id_index = get_id_index()
    models = get_zd_models()
    gate = get_gate()

    g = p.windows.gather_idx[frame_idx]
    x_raw = torch.from_numpy(X_raw[g])
    x_fus = torch.from_numpy(X_fus[g])
    st_t = torch.from_numpy(streams[frame_idx])
    mk_t = torch.from_numpy(mask[frame_idx])

    ex = gate.explain(residual[frame_idx], id_index[frame_idx],
                      streams[frame_idx], mask[frame_idx])
    gate_alarm = ex["score"] > gate.threshold

    out: Dict[str, Dict[str, np.ndarray]] = {"gate": {**ex, "alarm": gate_alarm,
                                                      "threshold": gate.threshold}}
    for name, model in models.items():
        if name == "TwinShield-Fusion":
            logits, alpha = model.forward_with_attention(x_fus, st_t, mk_t)
            out[name] = {"attention": alpha.numpy()}
        else:
            logits = model(x_raw)
            out[name] = {}
        proba = torch.softmax(logits, 1).numpy()
        pred = proba.argmax(1)
        sup_alarm = pred != 0                    # index 0 is Normal in the reduced space
        out[name].update({
            "proba": proba, "pred": pred, "conf": proba.max(1),
            "supervised_alarm": sup_alarm,
            "gate_alarm": gate_alarm if name == "TwinShield-Fusion"
                          else np.zeros_like(sup_alarm),
            "alarm": (sup_alarm | gate_alarm) if name == "TwinShield-Fusion" else sup_alarm,
        })
    return out


# ---------------------------------------------------------------------------
# Replay cases -- contiguous slices of the real capture
# ---------------------------------------------------------------------------
def _lead_range(p: Prepared, lead: int) -> Tuple[int, int]:
    """A run of genuinely normal frames from a held-out block.

    The quiet opening of the demo should also be data the models never trained
    on, otherwise "nothing fires yet" proves nothing.
    """
    y, split = p.windows.y, p.windows.split
    src = p.df["source"].to_numpy()
    blocks = p.windows.block_id
    normal_idx = np.where(src == "normal")[0]

    for b in np.unique(blocks[normal_idx]):
        pos = np.where((blocks == b) & (src == "normal"))[0]
        if len(pos) < lead:
            continue
        seg = slice(int(pos.min()), int(pos.min()) + lead)
        if (y[seg] == 0).all() and (split[seg] == "test").all():
            return seg.start, seg.stop
    start = max(0, int(normal_idx.max()) - lead)      # no fully held-out run available
    return start, start + lead


def _best_segment(p: Prepared, target: int, length: int, step: int = 8) -> int:
    """Start of the `length`-frame window carrying most frames of `target`.

    Ties break toward held-out data, so what the audience watches is as close to
    test data as the capture allows.
    """
    y, split = p.windows.y, p.windows.split
    attack_idx = np.where(p.df["source"].to_numpy() == "attack")[0]
    atk0, atk1 = int(attack_idx.min()), int(attack_idx.max())

    best, best_score = atk0, -1e9
    for start in range(atk0, max(atk0 + 1, atk1 - length), step):
        seg = slice(start, start + length)
        n_cls = int((y[seg] == target).sum())
        if n_cls == 0:
            continue
        score = 3.0 * min(n_cls, 40) / 40.0 + float((split[seg] == "test").mean())
        if score > best_score:
            best, best_score = start, score
    return best


@st.cache_data(show_spinner=False)
def get_demo_cases(lead: int = 60, length: int = 200) -> Dict[str, Dict]:
    """The two replay cases the Live Monitor compares all four models on.

    Case 1 is an attack every model was trained on.  Case 2 is the class
    `train_zeroday.py` quarantined, so for the three baselines it is genuinely
    an attack they have never seen.

    Both are assembled from two real recordings played back to back: a lead-in
    from the normal-driving capture, then a slice of the attack capture.  That
    join is necessary and is stated openly on the page -- in GEM-CAN the attack
    capture is already under attack from its first frame, so no stretch of it
    runs clean and then turns malicious.  No frame is altered or generated, and
    each is still classified from its own true preceding context.
    """
    p = get_prepared()
    y, split = p.windows.y, p.windows.split
    if not (p.df["source"].to_numpy() == "attack").any():
        return {}

    zd = get_zero_day() if zeroday_ready() else {"unknown_class": "Brake Tampering"}
    unknown_name = zd["unknown_class"]

    l0, l1 = _lead_range(p, lead)
    l_idx = np.arange(l0, l1)
    attack_idx = np.where(p.df["source"].to_numpy() == "attack")[0]
    atk1 = int(attack_idx.max())

    def build(key: str, title: str, target_name: str, known: bool) -> Optional[Dict]:
        target = CLASS_TO_IDX[target_name]
        if not (y == target).any():
            return None
        a0 = _best_segment(p, target, length)
        a_idx = np.arange(a0, min(a0 + length, atk1 + 1))
        idx = np.concatenate([l_idx, a_idx])
        return {
            "key": key, "title": title, "target": target_name, "known": known,
            "frames": idx, "lead": len(l_idx),
            "attack_range": (int(a_idx.min()), int(a_idx.max())),
            "lead_range": (l0, l1),
            "n_target": int((y[a_idx] == target).sum()),
            "test_share": round(float((split[idx] == "test").mean()) * 100, 1),
            "composition": {CLASS_NAMES[c]: int((y[idx] == c).sum())
                            for c in range(len(CLASS_NAMES)) if (y[idx] == c).any()},
        }

    cases: Dict[str, Dict] = {}
    known_name = next((c for c in CLASS_NAMES[1:]
                       if c != unknown_name and (y == CLASS_TO_IDX[c]).any()), None)
    if known_name:
        c = build("known", f"Test Case 1 — Known Attack · {known_name}", known_name, True)
        if c:
            cases[c["title"]] = c
    c = build("unknown", f"Test Case 2 — Unknown Attack · {unknown_name}",
              unknown_name, False)
    if c:
        cases[c["title"]] = c
    return cases


def frame_table(idx: np.ndarray, preds: np.ndarray, confs: np.ndarray,
                names: Optional[List[str]] = None,
                flagged: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Human-readable CAN frames, straight from the dataset.

    `names` is the label space the predictions came from -- the zero-day
    detectors emit a reduced one, so it cannot be assumed to be CLASS_NAMES.
    `flagged` marks frames the anomaly gate raised even though the supervised
    head said Normal; the table shows those as UNKNOWN ANOMALY, which is exactly
    what the system can honestly say about an attack it has no label for.
    """
    p = get_prepared()
    names = names or CLASS_NAMES
    df = p.df.iloc[idx]
    payload = df[DATA_COLS].to_numpy(dtype=int)
    labels = [names[c] for c in preds]
    if flagged is not None:
        labels = [("Unknown Anomaly" if (f and l == "Normal") else l)
                  for l, f in zip(labels, flagged)]
    return pd.DataFrame({
        "Time": df["timestamp"].to_numpy(),
        "CAN ID": df["arbitration_id_hex"].to_numpy(),
        "DLC": df["dlc"].to_numpy(),
        "Payload": [" ".join(f"{b:02X}" for b in row) for row in payload],
        "Predicted": labels,
        "Confidence": confs,
        "Actual": df["class_name"].to_numpy(),
    })


def id_stream_labels() -> List[str]:
    p = get_prepared()
    inv = {v: k for k, v in p.id_vocab.items()}
    return ["<unknown>"] + [f"{inv[i]:08X}" for i in range(1, len(p.id_vocab) + 1)]
