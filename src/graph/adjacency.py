from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def topk_mask(scores: torch.Tensor, top_k: int, directed: bool = True) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix")
    n = scores.shape[0]
    work = scores.clone()
    work.fill_diagonal_(float("-inf"))
    k = min(top_k, n - 1)
    mask = torch.zeros_like(work)
    if k > 0:
        idx = torch.topk(work, k=k, dim=1).indices
        mask.scatter_(1, idx, 1.0)
    if not directed:
        mask = torch.maximum(mask, mask.T)
    return mask


def identity_adjacency(num_nodes: int) -> torch.Tensor:
    return torch.eye(num_nodes, dtype=torch.float32)


def normalize_adjacency(adjacency: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    a = adjacency.float()
    if add_self_loops:
        a = a.clone()
        a.fill_diagonal_(0.0)
        a = a + torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    degree = a.sum(dim=1).clamp_min(1e-12)
    d_inv_sqrt = torch.pow(degree, -0.5)
    return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]


def correlation_adjacency(train_windows: np.ndarray, top_k: int, directed: bool = False) -> torch.Tensor:
    """Build a static graph from train-only residual/raw windows.

    Parameters
    ----------
    train_windows:
        Array with shape [samples, tickers, lookback]. Only the caller's train
        rows should be passed here; the function intentionally has no access to
        validation/test samples.
    """
    if train_windows.ndim != 3:
        raise ValueError(f"Expected [samples, tickers, lookback], got {train_windows.shape}")
    num_nodes = train_windows.shape[1]
    series = np.transpose(train_windows, (1, 0, 2)).reshape(num_nodes, -1)
    corr = np.corrcoef(series)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    scores = torch.tensor(corr, dtype=torch.float32)
    mask = topk_mask(scores, top_k=top_k, directed=directed)
    return scores * mask


def random_adjacency(num_nodes: int, top_k: int, seed: int, directed: bool = False) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    scores = torch.tensor(rng.random((num_nodes, num_nodes)), dtype=torch.float32)
    scores.fill_diagonal_(0.0)
    mask = topk_mask(scores, top_k=top_k, directed=directed)
    return scores * mask


@dataclass(frozen=True)
class AdjacencySpec:
    graph_type: str
    top_k: int | None = None
    directed: bool = False
    graph_seed: int | None = None


class LearnedStaticAdjacency(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, top_k: int, directed: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.directed = directed
        self.u = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.v = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def scores(self) -> torch.Tensor:
        s = F.relu(torch.tanh(self.u @ self.v.T))
        s = s.clone()
        s.fill_diagonal_(0.0)
        return s

    def forward(self) -> torch.Tensor:
        s = self.scores()
        mask = topk_mask(s, top_k=self.top_k, directed=self.directed)
        return s * mask


def build_fixed_adjacency(spec: AdjacencySpec, train_windows: np.ndarray, num_nodes: int) -> torch.Tensor | None:
    if spec.graph_type == "identity":
        return identity_adjacency(num_nodes)
    if spec.graph_type == "correlation":
        if spec.top_k is None:
            raise ValueError("correlation graph requires top_k")
        return correlation_adjacency(train_windows, top_k=spec.top_k, directed=spec.directed)
    if spec.graph_type == "random":
        if spec.top_k is None or spec.graph_seed is None:
            raise ValueError("random graph requires top_k and graph_seed")
        return random_adjacency(num_nodes, top_k=spec.top_k, seed=spec.graph_seed, directed=spec.directed)
    if spec.graph_type == "learned":
        return None
    if spec.graph_type == "none":
        return None
    raise ValueError(f"Unknown graph_type: {spec.graph_type}")

