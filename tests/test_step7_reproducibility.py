from src.graph.reproducibility import seed_everything
import torch


def test_step7_seed_reproducible_torch_draws():
    seed_everything(42)
    a = torch.randn(3)
    seed_everything(42)
    b = torch.randn(3)
    assert torch.allclose(a, b)
