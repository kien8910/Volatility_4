from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.graph.adjacency import topk_mask


class LearnedGraphBank(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, num_graphs: int, top_k: int, directed: bool = False):
        super().__init__()
        if num_graphs < 1:
            raise ValueError("num_graphs must be at least 1")
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.num_graphs = int(num_graphs)
        self.top_k = int(top_k)
        self.directed = bool(directed)
        self.u = nn.Parameter(torch.empty(num_graphs, num_nodes, embedding_dim))
        self.v = nn.Parameter(torch.empty(num_graphs, num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def dense_scores(self) -> torch.Tensor:
        scores = F.relu(torch.tanh(torch.matmul(self.u, self.v.transpose(-1, -2))))
        scores = scores.clone()
        diag = torch.arange(self.num_nodes, device=scores.device)
        scores[:, diag, diag] = 0.0
        return scores

    def topk_adjacency_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        rows = []
        for k in range(scores.shape[0]):
            mask = topk_mask(scores[k], top_k=self.top_k, directed=self.directed)
            rows.append(scores[k] * mask)
        return torch.stack(rows, dim=0)

    def forward(self) -> torch.Tensor:
        return self.topk_adjacency_from_scores(self.dense_scores())

