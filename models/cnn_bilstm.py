"""
CNN-BiLSTM detector.

Convolutions extract local bus patterns (bursts, byte co-changes); the BiLSTM
models how those patterns evolve across the window.  This is the backbone that
TwinShield-Fusion extends, so the ablation compares like with like.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTMBackbone(nn.Module):
    """Shared feature extractor -> returns a pooled embedding."""

    def __init__(self, n_features: int, channels: int = 64, hidden: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.out_dim = hidden * 2 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)      # [B, T, C]
        out, _ = self.lstm(h)
        return torch.cat([out[:, -1], out.mean(dim=1)], dim=1)


class CNNBiLSTMDetector(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = CNNBiLSTMBackbone(n_features)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
