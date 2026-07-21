import numpy as np

from src.stock_news_impact.spillover import graph_neighbor_weights


def test_graph_neighbor_spillover_uses_adjacency():
    adj = np.array([[0, 0.5], [0.2, 0]])
    weights = graph_neighbor_weights(adj, 0, max_hops=1)
    assert weights[0] == 1.0
    assert weights[1] == 0.5

