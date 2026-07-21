from __future__ import annotations

import torch
from torch import nn


class HierarchicalAdditiveFusion(nn.Module):
    def __init__(self, stock_dim: int, projection_dim: int, hidden_dims: list[int], num_horizons: int, dropout: float):
        super().__init__()
        self.stock_align = nn.Identity() if stock_dim == projection_dim else nn.Linear(stock_dim, projection_dim)
        layers: list[nn.Module] = []
        prev = projection_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = int(hidden)
        layers.append(nn.Linear(prev, num_horizons))
        self.head = nn.Sequential(*layers)

    def forward(self, stock_embedding: torch.Tensor, messages: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = self.stock_align(stock_embedding)
        for message in messages.values():
            fused = fused + message
        return self.head(fused)

