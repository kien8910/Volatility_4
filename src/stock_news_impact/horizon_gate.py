from __future__ import annotations

import torch
from torch import nn

from src.stock_news_impact.reliability_gate import logit


class HorizonGate(nn.Module):
    def __init__(self, input_dim: int, horizons: list[int], horizon_embedding_dim: int = 8, dropout: float = 0.1, initial_probability: float = 0.10):
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.horizon_to_idx = {h: i for i, h in enumerate(self.horizons)}
        self.embedding = nn.Embedding(len(self.horizons), horizon_embedding_dim)
        hidden = max(16, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + horizon_embedding_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.net[-1].bias, logit(initial_probability))

    def horizon_indices(self, horizons: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(horizons, dtype=torch.long)
        for h, pos in self.horizon_to_idx.items():
            idx = torch.where(horizons.long().eq(h), torch.full_like(idx, pos), idx)
        return idx

    def forward(self, x: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(self.horizon_indices(horizons))
        return torch.sigmoid(self.net(torch.cat([x, emb], dim=-1))).squeeze(-1)
