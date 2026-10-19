"""Hybrid CNN + LSTM + Transformer + GAT multi-task model.

Input: a sequence of L consecutive epochs from one recording
    x    (B, L, C, T)   raw EEG epochs (scaled)
    node (B, L, C, 5)   per-channel log band power (node features of the brain graph)
    adj  (B, L, C, C)   connectivity (PLV or coherence) = edge weights
Branches (each can be switched off in the config — needed for the A6-A10 ablation):
    CNN         EEGNet-style encoder per epoch -> e_t (B, L, E);   Z_cnn  = mean_t e_t
    LSTM        over e_1..e_L                                     Z_lstm = last hidden state
    Transformer 2 encoder layers + sinusoidal positions over e_t  Z_trans = mean over positions
    GAT         2-layer GAT per epoch graph, pooled over nodes, then mean over the sequence -> Z_gnn
Fusion: concat [enabled Z's] -> shared MLP -> 4 task heads (workload, attention, fatigue,
readiness). On STEW only the workload head has labels; the other heads are masked out of the
loss (see `masked_multitask_loss`) until team-collected labels exist.
All temporal branches accept any L >= 1, so the real-time engine can start before 8 epochs exist.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from eegengine.models.deep.eegnet import EEGNetEncoder
from eegengine.models.deep.gat import GATEncoder

ALL_BRANCHES = ("cnn", "lstm", "transformer", "gnn")
HEADS = ("workload", "attention", "fatigue", "readiness")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class HybridModel(nn.Module):
    def __init__(self, n_classes: int, branches=ALL_BRANCHES, n_channels: int = 14, n_samples: int = 512,
                 emb_dim: int = 64, cnn: dict | None = None, lstm_hidden: int = 64, trans_heads: int = 4,
                 trans_layers: int = 2, trans_ff: int = 128, gnn: dict | None = None, node_dim: int = 5,
                 fusion_dim: int = 64, dropout: float = 0.3, aux_classes: int = 2):
        super().__init__()
        self.branches = tuple(b for b in ALL_BRANCHES if b in branches)
        if not self.branches:
            raise ValueError("at least one branch must be enabled")
        self.uses_cnn = any(b in self.branches for b in ("cnn", "lstm", "transformer"))
        fused = 0
        if self.uses_cnn:
            self.cnn = EEGNetEncoder(n_channels, n_samples, emb_dim=emb_dim, **(cnn or {}))
        if "cnn" in self.branches:
            fused += emb_dim
        if "lstm" in self.branches:
            self.lstm = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)
            fused += lstm_hidden
        if "transformer" in self.branches:
            self.pos = PositionalEncoding(emb_dim)
            layer = nn.TransformerEncoderLayer(emb_dim, trans_heads, trans_ff, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, trans_layers)
            fused += emb_dim
        if "gnn" in self.branches:
            self.gnn = GATEncoder(in_dim=node_dim, **(gnn or {}))
            fused += self.gnn.out_dim
        self.fusion = nn.Sequential(nn.Linear(fused, fusion_dim), nn.ELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({
            "workload": nn.Linear(fusion_dim, n_classes),
            **{h: nn.Linear(fusion_dim, aux_classes) for h in HEADS[1:]},
        })

    def forward(self, x: torch.Tensor, node: torch.Tensor | None = None, adj: torch.Tensor | None = None):
        B, L = x.shape[:2]
        zs = []
        if self.uses_cnn:
            e = self.cnn(x.reshape(B * L, *x.shape[2:])).view(B, L, -1)   # per-epoch embeddings
            if "cnn" in self.branches:
                zs.append(e.mean(dim=1))
            if "lstm" in self.branches:
                _, (h, _) = self.lstm(e)
                zs.append(h[-1])
            if "transformer" in self.branches:
                zs.append(self.transformer(self.pos(e)).mean(dim=1))
        if "gnn" in self.branches:
            g = self.gnn(node.reshape(B * L, *node.shape[2:]), adj.reshape(B * L, *adj.shape[2:]))
            zs.append(g.view(B, L, -1).mean(dim=1))
        z = self.fusion(torch.cat(zs, dim=1))
        return {name: head(z) for name, head in self.heads.items()}


def masked_multitask_loss(outputs: dict, targets: dict, masks: dict, class_weights: dict | None = None,
                          head_weights: dict | None = None) -> torch.Tensor:
    """Sum over heads of the mean cross-entropy on the samples whose label exists (mask = True).

    A head with no labelled samples in the batch contributes exactly 0 (and no gradient), so the
    attention/fatigue/readiness heads are inert on STEW but ready for labelled team data.
    """
    total = outputs["workload"].new_zeros(())
    for name, logits in outputs.items():
        if name not in targets:
            continue
        m = masks[name]
        if m.sum() == 0:
            continue
        w = (class_weights or {}).get(name)
        loss = F.cross_entropy(logits[m], targets[name][m], weight=w)
        total = total + (head_weights or {}).get(name, 1.0) * loss
    return total
