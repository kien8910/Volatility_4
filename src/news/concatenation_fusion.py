from __future__ import annotations

import torch
from torch import nn


class EarlyConcatenationFusion(nn.Module):
    def __init__(self, stock_dim: int, projection_dim: int, control_dim: int, num_hierarchies: int, hidden_dims: list[int], num_horizons: int, dropout: float):
        super().__init__()
        in_dim = stock_dim + projection_dim * num_hierarchies + control_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = int(hidden)
        layers.append(nn.Linear(prev, num_horizons))
        self.net = nn.Sequential(*layers)

    def forward(self, stock_embedding: torch.Tensor, messages: dict[str, torch.Tensor], controls: torch.Tensor) -> torch.Tensor:
        ordered = [messages[key] for key in sorted(messages)]
        return self.net(torch.cat([stock_embedding, *ordered, controls], dim=-1))

