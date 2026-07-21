from __future__ import annotations

import torch
from torch import nn


class HeterogeneousRelationFusion(nn.Module):
    """A lightweight relation-specific message-passing layer without external graph deps."""

    def __init__(self, stock_dim: int, projection_dim: int, hidden_dims: list[int], num_horizons: int, dropout: float):
        super().__init__()
        self.stock_align = nn.Identity() if stock_dim == projection_dim else nn.Linear(stock_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)
        layers: list[nn.Module] = []
        prev = projection_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = int(hidden)
        layers.append(nn.Linear(prev, num_horizons))
        self.head = nn.Sequential(*layers)

    def forward(self, stock_embedding: torch.Tensor, messages: dict[str, torch.Tensor], static_adjacency: torch.Tensor | None = None) -> torch.Tensor:
        stock = self.stock_align(stock_embedding)
        if static_adjacency is not None:
            adj = static_adjacency.to(stock.device, stock.dtype)
            denom = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            stock = stock + torch.einsum("ij,bjd->bid", adj / denom, stock)
        relation_message = torch.stack(list(messages.values()), dim=0).sum(dim=0)
        return self.head(self.norm(stock + relation_message))

