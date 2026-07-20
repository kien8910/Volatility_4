import numpy as np
import torch

from src.graph.adjacency import correlation_adjacency, normalize_adjacency, random_adjacency


def test_correlation_graph_topk_uses_passed_train_rows_only():
    train = np.zeros((4, 3, 5), dtype=np.float32)
    train[:, 0, :] = np.arange(20).reshape(4, 5)
    train[:, 1, :] = train[:, 0, :]
    train[:, 2, :] = -train[:, 0, :]
    adj = correlation_adjacency(train, top_k=1, directed=True)
    assert adj.shape == (3, 3)
    assert torch.allclose(torch.diag(adj), torch.zeros(3))
    assert (adj > 0).sum(dim=1).max().item() == 1


def test_random_graph_density_matches_topk():
    adj = random_adjacency(num_nodes=11, top_k=3, seed=42, directed=True)
    assert adj.shape == (11, 11)
    assert int((adj > 0).sum().item()) == 33
    assert torch.allclose(torch.diag(adj), torch.zeros(11))


def test_graph_normalization_is_symmetric_for_symmetric_input():
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    norm = normalize_adjacency(adj, add_self_loops=True)
    assert torch.allclose(norm, norm.T)
    assert torch.isfinite(norm).all()

