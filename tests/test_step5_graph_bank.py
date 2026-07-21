import torch

from src.regime_graph.graph_bank import LearnedGraphBank


def test_graph_bank_shapes_and_topk():
    bank = LearnedGraphBank(num_nodes=5, embedding_dim=4, num_graphs=3, top_k=2, directed=True)
    adj = bank()
    assert adj.shape == (3, 5, 5)
    assert torch.allclose(torch.diagonal(adj, dim1=1, dim2=2), torch.zeros(3, 5))
    assert int((adj > 0).sum(dim=2).max().item()) <= 2


def test_graph_bank_k_one_valid():
    bank = LearnedGraphBank(num_nodes=4, embedding_dim=2, num_graphs=1, top_k=1)
    assert bank().shape == (1, 4, 4)

