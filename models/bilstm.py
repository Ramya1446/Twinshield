"""Bidirectional LSTM baseline over the context window."""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMDetector(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2 * 2, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                                # [B, T, 2H]
        pooled = torch.cat([out[:, -1], out.mean(dim=1)], dim=1)
        return self.head(pooled)
