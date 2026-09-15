"""
SHAP explanations for TwinShield-Fusion.

Uses shap.GradientExplainer against the trained PyTorch detector.  The model
takes two informative inputs -- the raw+residual context window and the per-ECU
residual streams -- so both are explained and reported:

    * feature attributions  -> which payload byte / timing / residual feature
                               pushed the model toward the predicted class
    * ECU stream attributions -> which ECU's residual mattered

Values are computed once (after training) and cached to results/shap.npz so the
demo loads instantly.  They are never hand-written.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SHAP_FILE = os.path.join(RESULTS, "shap.npz")


class _Wrapper(nn.Module):
    """Two-input view of TwinShieldFusion.

    The presence mask is recovered from the stream tensor itself (absent streams
    are exactly zero), so SHAP only has to perturb genuine inputs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        mask = (streams.abs().sum(-1) > 0).float()
        return self.model(x, streams, mask)


def compute_shap(
    model,
    X: np.ndarray,
    gather: np.ndarray,
    streams: np.ndarray,
    background_idx: np.ndarray,
    explain_idx: np.ndarray,
    feature_names: List[str],
    stream_labels: List[str],
    class_names: List[str],
    seed: int = 0,
    nsamples: int = 64,
) -> Dict[str, np.ndarray]:
    """Return SHAP attributions for the frames in `explain_idx`.

    `nsamples` is the number of gradient samples SHAP draws per explained frame.
    The library default (200) means 200 forward+backward passes through the
    CNN-BiLSTM for every frame, which runs for hours on CPU at this sample size;
    64 keeps the ranking stable while finishing in minutes.
    """
    import shap

    torch.manual_seed(seed)
    wrapper = _Wrapper(model).eval()

    bg_x = torch.from_numpy(X[gather[background_idx]])
    bg_s = torch.from_numpy(streams[background_idx])
    ex_x = torch.from_numpy(X[gather[explain_idx]])
    ex_s = torch.from_numpy(streams[explain_idx])

    explainer = shap.GradientExplainer(wrapper, [bg_x, bg_s])
    values = explainer.shap_values([ex_x, ex_s], nsamples=nsamples)

    # shap returns either a list per output, or a stacked array with a trailing
    # class axis, depending on version -- normalise to [n_inputs][N, ..., C].
    if isinstance(values, list) and len(values) and isinstance(values[0], list):
        sv_x = np.stack(values[0], axis=-1)
        sv_s = np.stack(values[1], axis=-1)
    else:
        sv_x, sv_s = values[0], values[1]
        if sv_x.ndim == 3:      # no class axis (single output)
            sv_x = sv_x[..., None]
            sv_s = sv_s[..., None]

    with torch.no_grad():
        pred = wrapper(ex_x, ex_s).argmax(1).numpy()

    n = len(explain_idx)
    n_cls = sv_x.shape[-1]
    # Attribution toward each frame's own predicted class.
    take = np.minimum(pred, n_cls - 1)
    feat = np.stack([sv_x[i, ..., take[i]].sum(axis=0) for i in range(n)])   # [N, F]
    ecu = np.stack([sv_s[i, ..., take[i]].sum(axis=-1) for i in range(n)])   # [N, S]

    return {
        "feature_shap": feat.astype(np.float32),
        "ecu_shap": ecu.astype(np.float32),
        "pred": pred.astype(np.int64),
        "explain_idx": explain_idx.astype(np.int64),
        "feature_names": np.array(feature_names, dtype=object),
        "stream_labels": np.array(stream_labels, dtype=object),
        "class_names": np.array(class_names, dtype=object),
    }


def save(payload: Dict[str, np.ndarray]) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(SHAP_FILE, **payload)
    return SHAP_FILE


def load() -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(SHAP_FILE):
        return None
    z = np.load(SHAP_FILE, allow_pickle=True)
    return {k: z[k] for k in z.files}


def aggregate(payload: Dict[str, np.ndarray], class_idx: Optional[int] = None,
              top_k: int = 12) -> Tuple[List[str], np.ndarray]:
    """Mean absolute attribution per feature, optionally restricted to one class."""
    feat = payload["feature_shap"]
    names = list(payload["feature_names"])
    if class_idx is not None:
        sel = payload["pred"] == class_idx
        if sel.sum() == 0:
            return [], np.array([])
        feat = feat[sel]
    imp = np.abs(feat).mean(axis=0)
    order = np.argsort(imp)[::-1][:top_k]
    return [names[i] for i in order], imp[order]
