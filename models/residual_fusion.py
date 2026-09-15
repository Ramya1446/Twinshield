"""
Adaptive Multi-ECU Residual Fusion.

Every arbitration ID on the bus is treated as a separate ECU *stream*.  For a
given context window we summarise each stream by its mean Digital-Twin residual
across the 8 payload bytes, giving a tensor [B, n_streams, 8].  Streams absent
from the window are masked out.

An additive (Bahdanau-style) attention layer then learns how much each stream
should contribute to the decision for this particular window:

    e_i     = v^T tanh(W h_i + b)
    alpha_i = softmax(e_i)   over streams present in the window
    fused   = sum_i alpha_i * h_i

The point is adaptivity: a DoS burst concentrates evidence on one stream, while
brake tampering concentrates it on the brake ID's residual.  A fixed average
over streams would dilute both.  The alpha_i are returned so the UI can show
which ECU stream actually drove a detection -- they are real learnt weights,
not decoration.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .cnn_bilstm import CNNBiLSTMBackbone


class AdaptiveECUAttention(nn.Module):
    def __init__(self, stream_dim: int = 8, hidden: int = 48, out_dim: int = 48):
        super().__init__()
        self.proj = nn.Linear(stream_dim, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.out = nn.Sequential(nn.Linear(stream_dim, out_dim), nn.ReLU())

    def forward(
        self, streams: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """streams [B, S, D], mask [B, S] (1 = stream present in window)."""
        e = self.score(torch.tanh(self.proj(streams))).squeeze(-1)      # [B, S]
        e = e.masked_fill(mask <= 0, torch.finfo(e.dtype).min)
        alpha = torch.softmax(e, dim=1)
        # A window with no streams at all would give a uniform-but-meaningless
        # alpha; zero it so the fused vector stays neutral.
        alpha = alpha * (mask.sum(dim=1, keepdim=True) > 0).float()
        fused = torch.bmm(alpha.unsqueeze(1), streams).squeeze(1)       # [B, D]
        return self.out(fused), alpha


class TwinShieldFusion(nn.Module):
    """Proposed model: CNN-BiLSTM over (raw + residual) + adaptive ECU fusion."""

    def __init__(self, n_features: int, n_classes: int, n_streams: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = CNNBiLSTMBackbone(n_features)
        self.fusion = AdaptiveECUAttention()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim + 48, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.n_streams = n_streams

    def forward(
        self, x: torch.Tensor, streams: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        fused, self._alpha = self.fusion(streams, mask)
        return self.head(torch.cat([self.backbone(x), fused], dim=1))

    def forward_with_attention(
        self, x: torch.Tensor, streams: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused, alpha = self.fusion(streams, mask)
        logits = self.head(torch.cat([self.backbone(x), fused], dim=1))
        return logits, alpha
