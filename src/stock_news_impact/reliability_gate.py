from __future__ import annotations

import math

import torch
from torch import nn


def logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class ReliabilityGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1, initial_probability: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.net[-1].bias, logit(initial_probability))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)
