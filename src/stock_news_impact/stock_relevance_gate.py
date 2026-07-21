from __future__ import annotations

import torch
from torch import nn

from src.stock_news_impact.reliability_gate import logit


class StockRelevanceGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1, initial_probability: float = 0.10):
        super().__init__()
        dims = [input_dim] + [int(x) for x in hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        nn.init.constant_(self.net[-1].bias, logit(initial_probability))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)
