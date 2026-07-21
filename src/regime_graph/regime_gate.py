from __future__ import annotations

import torch
from torch import nn


class RegimeGate(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_regimes: int, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_regimes),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        logits = self.net(state_features) / self.temperature
        return torch.softmax(logits, dim=-1)

