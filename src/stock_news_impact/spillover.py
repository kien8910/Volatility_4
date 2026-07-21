from __future__ import annotations

import numpy as np


def graph_neighbor_weights(adjacency: np.ndarray, source_index: int, max_hops: int = 1) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=float)
    weights = np.zeros(adj.shape[0], dtype=float)
    frontier = np.zeros(adj.shape[0], dtype=bool)
    frontier[int(source_index)] = True
    visited = frontier.copy()
    for _ in range(int(max_hops)):
        next_frontier = (adj[frontier].sum(axis=0) > 0) & ~visited
        weights[next_frontier] = adj[int(source_index), next_frontier]
        visited |= next_frontier
        frontier = next_frontier
    weights[int(source_index)] = 1.0
    return weights
