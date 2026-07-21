import torch

from src.graph.reproducibility import seed_everything
from src.regime_graph.graph_bank import LearnedGraphBank


def test_same_seed_reproduces_graph_initialization():
    seed_everything(42)
    a = LearnedGraphBank(4, 3, 1, 2).dense_scores()
    seed_everything(42)
    b = LearnedGraphBank(4, 3, 1, 2).dense_scores()
    assert torch.allclose(a, b)

