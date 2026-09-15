"""
GRU Digital Twin.

Learns the *normal* behaviour of every ECU stream: given the previous k
payloads observed on a given arbitration ID, predict the next payload on that
ID.  Trained exclusively on Normal frames from the training split, so it never
sees an attack during fitting.

    Residual_t = | Actual_t - Predicted_t |        (per payload byte)

Under normal traffic the residual is small; when an attacker injects a forged
payload on a legitimate ID the observed bytes stop matching the learnt
behaviour and the residual rises.  This is the only mechanism in the system
that can flag tampering on an ID that also carries genuine traffic.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GRUDigitalTwin(nn.Module):
    def __init__(self, n_ids: int, hidden: int = 96, emb: int = 16, layers: int = 1):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, emb)
        self.gru = nn.GRU(8 + emb, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + emb, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),          # payload bytes are scaled to [0, 1]
        )

    def forward(self, payload_hist: torch.Tensor, id_idx: torch.Tensor) -> torch.Tensor:
        """payload_hist [B, k, 8] in [0,1];  id_idx [B] long -> predicted [B, 8]."""
        e = self.id_emb(id_idx)                       # [B, emb]
        e_seq = e.unsqueeze(1).expand(-1, payload_hist.size(1), -1)
        out, _ = self.gru(torch.cat([payload_hist, e_seq], dim=-1))
        return self.head(torch.cat([out[:, -1], e], dim=-1))


@torch.no_grad()
def compute_residuals(
    model: GRUDigitalTwin,
    payload: np.ndarray,       # [N, 8] scaled 0..1
    id_hist: np.ndarray,       # [N, k]
    id_idx: np.ndarray,        # [N]
    batch: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Per-frame |actual - predicted| for every frame in the dataset."""
    model.eval()
    n = len(payload)
    res = np.zeros((n, 8), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        hist = torch.from_numpy(payload[id_hist[s:e]]).float().to(device)
        ids = torch.from_numpy(id_idx[s:e]).long().to(device)
        pred = model(hist, ids).cpu().numpy()
        res[s:e] = np.abs(payload[s:e] - pred)
    return res


@torch.no_grad()
def predict_payload(
    model: GRUDigitalTwin,
    payload: np.ndarray,
    id_hist: np.ndarray,
    id_idx: np.ndarray,
    batch: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Expected next payload -- used by the Digital Twin page's Expected vs Actual plot."""
    model.eval()
    n = len(payload)
    out = np.zeros((n, 8), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        hist = torch.from_numpy(payload[id_hist[s:e]]).float().to(device)
        ids = torch.from_numpy(id_idx[s:e]).long().to(device)
        out[s:e] = model(hist, ids).cpu().numpy()
    return out
