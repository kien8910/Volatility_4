from __future__ import annotations

import torch
from torch import nn


class HierarchyProjection(nn.Module):
    def __init__(self, hierarchies: list[str], embedding_dim: int, projection_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hierarchies = list(hierarchies)
        self.layers = nn.ModuleDict(
            {
                hierarchy: nn.Sequential(
                    nn.Linear(embedding_dim, projection_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                )
                for hierarchy in self.hierarchies
            }
        )

    def forward(self, news_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        if news_embeddings.ndim != 4:
            raise ValueError("news_embeddings must have shape [batch, nodes, hierarchies, embedding_dim]")
        if news_embeddings.shape[2] != len(self.hierarchies):
            raise ValueError("news_embeddings hierarchy axis does not match configured hierarchies")
        return {
            hierarchy: self.layers[hierarchy](news_embeddings[:, :, idx, :])
            for idx, hierarchy in enumerate(self.hierarchies)
        }

