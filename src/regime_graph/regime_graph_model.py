from __future__ import annotations

import torch
from torch import nn

from src.graph.adjacency import normalize_adjacency
from src.graph.temporal_encoder import build_temporal_encoder
from src.regime_graph.ema_graph import EMAGraphBuffer
from src.regime_graph.graph_bank import LearnedGraphBank
from src.regime_graph.regime_gate import RegimeGate


class RegimeGraphForecastModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        lookback: int,
        num_horizons: int,
        temporal_cfg: dict,
        graph_embedding_dim: int,
        top_k: int,
        directed: bool,
        num_graphs: int,
        model_id: str,
        ema_beta: float = 0.0,
        state_dim: int = 7,
        gate_hidden_dim: int = 16,
        gate_temperature: float = 1.0,
        gate_dropout: float = 0.1,
        residual_lambda: float = 1.0,
        use_ema_for_training: bool = False,
        use_ema_for_validation: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.model_id = model_id
        self.ema_beta = float(ema_beta)
        self.residual_lambda = float(residual_lambda)
        self.use_ema_for_training = bool(use_ema_for_training)
        self.use_ema_for_validation = bool(use_ema_for_validation)
        hidden_dim = int(temporal_cfg["hidden_dim"])
        self.temporal = build_temporal_encoder("small_tcn", lookback, temporal_cfg)
        self.graph_bank = LearnedGraphBank(num_nodes, graph_embedding_dim, num_graphs, top_k, directed=directed)
        self.ema = EMAGraphBuffer((num_graphs, num_nodes, num_nodes), beta=ema_beta)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_horizons))
        self.gate = None
        if num_graphs > 1:
            self.gate = RegimeGate(state_dim, gate_hidden_dim, num_graphs, temperature=gate_temperature, dropout=gate_dropout)

    def current_dense_scores(self) -> torch.Tensor:
        return self.graph_bank.dense_scores()

    @torch.no_grad()
    def update_ema(self) -> torch.Tensor:
        return self.ema.update(self.current_dense_scores())

    def _scores_for_forward(self) -> torch.Tensor:
        current = self.current_dense_scores()
        use_ema = (self.training and self.use_ema_for_training) or ((not self.training) and self.use_ema_for_validation)
        if use_ema:
            return self.ema.value_or_current(current)
        return current

    def adjacency_bank(self) -> torch.Tensor:
        scores = self._scores_for_forward()
        return self.graph_bank.topk_adjacency_from_scores(scores)

    def regime_weights(self, state_features: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.num_graphs == 1:
            return torch.ones((batch_size, 1), device=device, dtype=dtype)
        if state_features is None:
            raise ValueError("state_features are required when num_graphs > 1")
        return self.gate(state_features)

    def forward(self, x: torch.Tensor, state_features: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.temporal(x)
        adj_bank = self.adjacency_bank().to(device=x.device, dtype=x.dtype)
        messages = []
        for k in range(self.num_graphs):
            a_norm = normalize_adjacency(adj_bank[k], add_self_loops=True).to(device=x.device, dtype=x.dtype)
            msg = torch.einsum("ij,bjd->bid", a_norm, h)
            messages.append(self.activation(self.message(msg)))
        stacked = torch.stack(messages, dim=1)
        weights = self.regime_weights(state_features, x.shape[0], x.device, x.dtype)
        mixed = torch.einsum("bk,bknd->bnd", weights, stacked)
        out = h + self.residual_lambda * mixed
        prediction = self.head(out)
        return prediction, {"regime_weights": weights, "adjacency_bank": adj_bank}

