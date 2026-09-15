"""1-D CNN baseline: temporal convolution over the context window."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNDetector(nn.Module):
    def __init__(self, n_features: int, n_classes: int, channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels * 2 * 2, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [B, T, F]
        h = self.conv(x.transpose(1, 2))                     # [B, C, T]
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        return self.head(pooled)
