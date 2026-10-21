"""PyTorch GRU/LSTM sequence classifier with padding masks."""
from __future__ import annotations

import torch
from torch import nn


class TemporalClassifier(nn.Module):
    """Input (B,T,F) + mask (B,T) -> logits (B,C).

    The representation concatenates the masked mean of RNN outputs and the output at the
    last valid timestep, so padded steps never influence the prediction.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64, num_layers: int = 1,
                 cell: str = "gru", dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.config = dict(n_features=n_features, n_classes=n_classes, hidden_size=hidden_size,
                           num_layers=num_layers, cell=cell, dropout=dropout, bidirectional=bidirectional)
        self.rnn = rnn_cls(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * out_dim, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B,T,F), got {tuple(x.shape)}")
        if x.shape[-1] != self.config["n_features"]:
            raise ValueError(f"Expected {self.config['n_features']} features, got {x.shape[-1]}")
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device)
        x = torch.nan_to_num(x, nan=0.0) * mask.unsqueeze(-1)
        out, _ = self.rnn(x)                               # (B,T,H)
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        last_idx = (mask.sum(1).long() - 1).clamp(min=0)
        last = out[torch.arange(out.shape[0], device=x.device), last_idx]
        return self.head(torch.cat([mean, last], dim=1))


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")
