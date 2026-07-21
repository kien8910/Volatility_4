from __future__ import annotations

import torch
from torch import nn

from src.stock_news_impact.horizon_gate import HorizonGate
from src.stock_news_impact.reliability_gate import ReliabilityGate
from src.stock_news_impact.stock_relevance_gate import StockRelevanceGate


class FactorizedGateModel(nn.Module):
    def __init__(
        self,
        event_dim: int,
        relation_dim: int,
        stock_dim: int,
        market_dim: int,
        horizons: list[int],
        reliability_hidden_dim: int = 32,
        relevance_hidden_dims: list[int] | None = None,
        horizon_embedding_dim: int = 8,
        dropout: float = 0.1,
        initial_probability: float = 0.10,
    ):
        super().__init__()
        relevance_hidden_dims = relevance_hidden_dims or [64, 32]
        self.reliability = ReliabilityGate(event_dim, reliability_hidden_dim, dropout, initial_probability)
        relevance_dim = event_dim + relation_dim + stock_dim + market_dim
        self.relevance = StockRelevanceGate(relevance_dim, relevance_hidden_dims, dropout, initial_probability)
        self.horizon = HorizonGate(relevance_dim, horizons, horizon_embedding_dim, dropout, initial_probability)

    def forward(
        self,
        event_features: torch.Tensor,
        relation_features: torch.Tensor,
        stock_features: torch.Tensor,
        market_features: torch.Tensor,
        horizons: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rel_input = torch.cat([event_features, relation_features, stock_features, market_features], dim=-1)
        reliability_gate = self.reliability(event_features)
        stock_relevance_gate = self.relevance(rel_input)
        horizon_gate = self.horizon(rel_input, horizons)
        final_gate = reliability_gate * stock_relevance_gate * horizon_gate
        return {
            "reliability_gate": reliability_gate,
            "stock_relevance_gate": stock_relevance_gate,
            "horizon_gate": horizon_gate,
            "final_gate": final_gate,
        }


class FixedGateModel(nn.Module):
    def __init__(self, gate_value: float):
        super().__init__()
        self.gate_value = float(gate_value)

    def forward(self, event_features, relation_features, stock_features, market_features, horizons):
        gate = torch.full((event_features.shape[0],), self.gate_value, dtype=event_features.dtype, device=event_features.device)
        return {
            "reliability_gate": gate,
            "stock_relevance_gate": gate,
            "horizon_gate": gate,
            "final_gate": gate,
        }
