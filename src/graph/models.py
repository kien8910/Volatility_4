from __future__ import annotations

import torch
from torch import nn

from src.graph.adjacency import LearnedStaticAdjacency
from src.graph.graph_layers import GCNLayer
from src.graph.temporal_encoder import build_temporal_encoder


class StaticGraphForecastModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        lookback: int,
        num_horizons: int,
        temporal_kind: str,
        temporal_cfg: dict,
        graph_type: str,
        fixed_adjacency: torch.Tensor | None = None,
        graph_embedding_dim: int | None = None,
        top_k: int | None = None,
        directed: bool = False,
        residual_lambda: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.lookback = lookback
        self.num_horizons = num_horizons
        self.graph_type = graph_type
        hidden_dim = int(temporal_cfg["hidden_dim"])
        self.temporal = build_temporal_encoder(temporal_kind, lookback, temporal_cfg)
        self.learned_adjacency = None
        if graph_type == "learned":
            if graph_embedding_dim is None or top_k is None:
                raise ValueError("learned graph requires graph_embedding_dim and top_k")
            self.learned_adjacency = LearnedStaticAdjacency(num_nodes, graph_embedding_dim, top_k, directed=directed)
            self.register_buffer("fixed_adjacency", torch.empty(0), persistent=False)
        elif fixed_adjacency is not None:
            self.register_buffer("fixed_adjacency", fixed_adjacency.float())
        else:
            self.register_buffer("fixed_adjacency", torch.empty(0), persistent=False)
        self.gcn = GCNLayer(hidden_dim=hidden_dim, residual_lambda=residual_lambda)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_horizons))

    def adjacency(self) -> torch.Tensor | None:
        if self.graph_type == "none":
            return None
        if self.learned_adjacency is not None:
            return self.learned_adjacency()
        if self.fixed_adjacency.numel() == 0:
            return None
        return self.fixed_adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)
        h = self.gcn(h, self.adjacency())
        return self.head(h)


class MaskedReconstructionModel(nn.Module):
    """Reconstruct the last residual state of masked tickers from cross-stock context."""

    def __init__(
        self,
        num_nodes: int,
        lookback: int,
        temporal_kind: str,
        temporal_cfg: dict,
        graph_type: str,
        fixed_adjacency: torch.Tensor | None = None,
        graph_embedding_dim: int | None = None,
        top_k: int | None = None,
        directed: bool = False,
        residual_lambda: float = 1.0,
    ):
        super().__init__()
        hidden_dim = int(temporal_cfg["hidden_dim"])
        self.graph_type = graph_type
        self.temporal = build_temporal_encoder(temporal_kind, lookback, temporal_cfg)
        self.learned_adjacency = None
        if graph_type == "learned":
            if graph_embedding_dim is None or top_k is None:
                raise ValueError("learned graph requires graph_embedding_dim and top_k")
            self.learned_adjacency = LearnedStaticAdjacency(num_nodes, graph_embedding_dim, top_k, directed=directed)
            self.register_buffer("fixed_adjacency", torch.empty(0), persistent=False)
        elif fixed_adjacency is not None:
            self.register_buffer("fixed_adjacency", fixed_adjacency.float())
        else:
            self.register_buffer("fixed_adjacency", torch.empty(0), persistent=False)
        self.gcn = GCNLayer(hidden_dim=hidden_dim, residual_lambda=residual_lambda)
        self.head = nn.Linear(hidden_dim, 1)

    def adjacency(self) -> torch.Tensor | None:
        if self.learned_adjacency is not None:
            return self.learned_adjacency()
        if self.fixed_adjacency.numel() == 0:
            return None
        return self.fixed_adjacency

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_x = x.masked_fill(mask[:, :, None], 0.0)
        h = self.temporal(masked_x)
        h = self.gcn(h, self.adjacency())
        return self.head(h).squeeze(-1)

