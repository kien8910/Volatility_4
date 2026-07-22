from __future__ import annotations
import math
import torch
from torch import nn


def segment_topk_mask(scores: torch.Tensor, row_index: torch.Tensor, k: int) -> torch.Tensor:
    mask = torch.zeros_like(scores)
    for row in torch.unique(row_index, sorted=True):
        positions = torch.nonzero(row_index.eq(row), as_tuple=False).flatten()
        keep = positions[torch.topk(scores[positions], min(int(k), len(positions)), sorted=False).indices]
        mask[keep] = 1.0
    return mask


class SparseHurdleCorrector(nn.Module):
    def __init__(self, embedding_dim: int, state_dim: int, tickers: int, horizons: tuple[int, ...], cfg: dict):
        super().__init__(); hidden = int(cfg["model"]["hidden_dim"])
        stock_dim = int(cfg["model"]["stock_embedding_dim"]); horizon_dim = int(cfg["model"]["horizon_embedding_dim"])
        type_dim = int(cfg["model"].get("event_type_embedding_dim", 4))
        self.horizons = tuple(int(x) for x in horizons); self.horizon_to_index = {h: i for i, h in enumerate(self.horizons)}
        self.event_projection = nn.Linear(embedding_dim, hidden, bias=False)
        self.event_type_embedding = nn.Embedding(int(cfg["model"].get("event_type_count", 16)), type_dim)
        self.stock_embedding = nn.Embedding(tickers, stock_dim)
        self.horizon_embedding = nn.Embedding(len(horizons), horizon_dim)
        edge_input = hidden + type_dim + 5 + state_dim + stock_dim + horizon_dim
        self.edge_gate = nn.Sequential(nn.Linear(edge_input, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        event_repr_dim = hidden + type_dim + 5
        row_input = event_repr_dim + state_dim + stock_dim + horizon_dim
        self.hurdle_head = nn.Sequential(nn.Linear(row_input, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.correction_head = nn.Sequential(nn.Linear(row_input, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.log_scale = nn.Parameter(torch.zeros(len(horizons)))
        alpha_max = float(cfg["model"]["alpha_max"]); alpha_init = float(cfg["model"]["alpha_init"])
        self.alpha_max = alpha_max
        ratio = min(max(alpha_init / alpha_max, 1e-5), 1 - 1e-5)
        self.alpha_logit = nn.Parameter(torch.full((len(horizons),), math.log(ratio / (1 - ratio))))
        gate_p = float(cfg["model"].get("edge_gate_init_probability", 0.10))
        hurdle_p = float(cfg["model"].get("hurdle_init_probability", 0.05))
        nn.init.constant_(self.edge_gate[-1].bias, math.log(gate_p / (1 - gate_p)))
        nn.init.constant_(self.hurdle_head[-1].bias, math.log(hurdle_p / (1 - hurdle_p)))
        nn.init.zeros_(self.correction_head[-1].weight)

    def horizon_index(self, horizon: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(horizon, -1, dtype=torch.long)
        for h, idx in self.horizon_to_index.items(): result[horizon.eq(h)] = idx
        if bool(result.lt(0).any()): raise ValueError("Encountered horizon not configured for sparse target-text pilot")
        return result

    def forward(self, embedding: torch.Tensor, event_meta: torch.Tensor, event_type: torch.Tensor,
                edge_row: torch.Tensor, row_state: torch.Tensor, row_stock: torch.Tensor,
                row_horizon: torch.Tensor, selection_mode: str, top_k: int) -> dict[str, torch.Tensor]:
        n_rows = len(row_state); hi = self.horizon_index(row_horizon)
        event_z = torch.tanh(self.event_projection(embedding))
        type_z = self.event_type_embedding(event_type)
        event_repr = torch.cat([event_z, type_z, event_meta], dim=-1)
        stock_z = self.stock_embedding(row_stock); horizon_z = self.horizon_embedding(hi)
        edge_context = torch.cat([event_repr, row_state[edge_row], stock_z[edge_row], horizon_z[edge_row]], dim=-1)
        logits = self.edge_gate(edge_context).squeeze(-1)
        if selection_mode == "learned_topk":
            selected_mask = segment_topk_mask(logits, edge_row, top_k)
            edge_gate = torch.sigmoid(logits) * selected_mask
        elif selection_mode in {"all", "deterministic_topk"}:
            selected_mask = torch.ones_like(logits)
            edge_gate = torch.ones_like(logits)
        else:
            raise ValueError(f"Unknown selection mode: {selection_mode}")
        counts = torch.zeros(n_rows, device=embedding.device).index_add_(0, edge_row, selected_mask)
        has_event = counts.gt(0)
        aggregated = torch.zeros((n_rows, event_repr.shape[1]), device=embedding.device)
        aggregated.index_add_(0, edge_row, event_repr * edge_gate[:, None])
        aggregated = aggregated / counts.clamp_min(1)[:, None]
        row_context = torch.cat([aggregated, row_state, stock_z, horizon_z], dim=-1)
        hurdle = torch.sigmoid(self.hurdle_head(row_context).squeeze(-1)) * has_event
        raw = self.correction_head(row_context).squeeze(-1)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit[hi])
        correction = alpha * hurdle * torch.tanh(raw / self.log_scale[hi].exp().clamp_min(1e-3)) * has_event
        return {"correction": correction, "hurdle_probability": hurdle,
                "edge_gate": edge_gate, "edge_logits": logits, "selected_mask": selected_mask, "has_event": has_event,
                "alpha": alpha}
