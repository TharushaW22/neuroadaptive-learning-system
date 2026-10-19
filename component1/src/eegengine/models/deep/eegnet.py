"""EEGNet (Lawhern et al., 2018, J. Neural Eng.) and the EEGNet-style epoch encoder.

EEGNet blocks for an input epoch of shape (1, channels, samples):
  1. temporal conv  (F1 filters, kernel = fs/2 samples)      -> learns band-pass filters
  2. depthwise spatial conv over all channels (D per filter)  -> learns spatial filters (like CSP)
     BatchNorm + ELU + AvgPool(4) + Dropout
  3. separable conv = depthwise temporal (16) + pointwise     -> combines the feature maps
     BatchNorm + ELU + AvgPool(8) + Dropout
  4. flatten -> linear classifier
The same blocks (without the classifier) are the per-epoch CNN encoder of the hybrid model.
"""

from __future__ import annotations

import torch
from torch import nn


class EEGNetEncoder(nn.Module):
    """Epoch (B, C, T) -> embedding (B, emb_dim)."""

    def __init__(self, n_channels: int = 14, n_samples: int = 512, F1: int = 8, D: int = 2, F2: int = 16,
                 kernel: int = 64, dropout: float = 0.25, emb_dim: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),   # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise temporal
            nn.Conv2d(F1 * D, F2, 1, bias=False),                                            # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_flat = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_samples))).numel()
        self.n_flat = n_flat
        self.proj = nn.Linear(n_flat, emb_dim) if emb_dim else nn.Identity()
        self.emb_dim = emb_dim or n_flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block2(self.block1(x.unsqueeze(1)))
        return self.proj(z.flatten(1))


class EEGNet(nn.Module):
    """Single-epoch classifier (the published EEGNet: flatten -> linear, no extra projection)."""

    def __init__(self, n_classes: int, n_channels: int = 14, n_samples: int = 512, F1: int = 8, D: int = 2,
                 F2: int = 16, kernel: int = 64, dropout: float = 0.25):
        super().__init__()
        self.encoder = EEGNetEncoder(n_channels, n_samples, F1, D, F2, kernel, dropout, emb_dim=0)
        self.classifier = nn.Linear(self.encoder.n_flat, n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, C, T) -> {"workload": logits (B, n_classes)}"""
        return {"workload": self.classifier(self.encoder(x))}
