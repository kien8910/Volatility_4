from __future__ import annotations

import torch
from torch import nn

from src.graph.adjacency import normalize_adjacency


class GCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, residual_lambda: float = 1.0, activation: str = "relu"):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.residual_lambda = residual_lambda
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
        if adjacency is None:
            return h
        a_norm = normalize_adjacency(adjacency.to(device=h.device, dtype=h.dtype), add_self_loops=True)
        msg = torch.einsum("ij,bjd->bid", a_norm, h)
        msg = self.activation(self.linear(msg))
        return h + self.residual_lambda * msg

