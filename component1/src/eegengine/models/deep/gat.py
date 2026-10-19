"""Graph Attention Network layer in plain PyTorch (no torch_geometric needed).

The brain graph is small and dense (14 electrodes), so adjacency is a dense (B, N, N) tensor.
For node i, GAT computes attention over its neighbours j:
    e_ij  = LeakyReLU( a_src . W h_i  +  a_dst . W h_j )  +  log(A_ij)      (edge weight as a prior)
    alpha = softmax_j(e_ij) over edges that exist (A_ij > 0, self-loops always kept)
    h_i'  = sum_j alpha_ij W h_j        (one result per head; heads concatenated or averaged)
Adding log(A_ij) lets strongly phase-locked electrode pairs get more attention a priori, while
the learned part decides which connections matter for the task (Velickovic et al., 2018).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h: (B, N, in_dim), adj: (B, N, N) non-negative weights -> (B, N, heads*out | out)."""
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.out_dim)                 # (B, N, H, D)
        src = (Wh * self.a_src).sum(-1)                                      # (B, N, H)
        dst = (Wh * self.a_dst).sum(-1)
        e = F.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), 0.2)           # (B, N, N, H)
        eye = torch.eye(N, device=h.device, dtype=adj.dtype).unsqueeze(0)
        A = adj * (1 - eye) + eye                                            # self-loops weight 1
        e = e + torch.log(A.clamp_min(1e-6)).unsqueeze(-1)                   # edge-weight prior
        e = e.masked_fill((A <= 0).unsqueeze(-1), float("-inf"))             # only existing edges
        alpha = self.dropout(torch.softmax(e, dim=2))                        # over neighbours j
        out = torch.einsum("bijh,bjhd->bihd", alpha, Wh)                     # (B, N, H, D)
        return out.reshape(B, N, -1) if self.concat else out.mean(dim=2)


class GATEncoder(nn.Module):
    """Two GAT layers + mean pooling over nodes: graph -> embedding (B, out_dim)."""

    def __init__(self, in_dim: int = 5, hidden: int = 16, out_dim: int = 32, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden, heads, concat=True, dropout=dropout)
        self.gat2 = GATLayer(hidden * heads, out_dim, heads, concat=False, dropout=dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.gat1(x, adj))
        h = F.elu(self.gat2(h, adj))
        return h.mean(dim=1)
