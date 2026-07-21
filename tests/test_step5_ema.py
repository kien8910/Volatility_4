import torch

from src.regime_graph.ema_graph import EMAGraphBuffer


def test_ema_beta_zero_matches_current():
    ema = EMAGraphBuffer((1, 2, 2), beta=0.0)
    first = torch.ones(1, 2, 2)
    second = torch.full((1, 2, 2), 3.0)
    ema.update(first)
    out = ema.update(second)
    assert torch.allclose(out, second)
    assert not ema.ema_score.requires_grad


def test_ema_update_formula():
    ema = EMAGraphBuffer((1, 1, 2), beta=0.5)
    ema.update(torch.tensor([[[1.0, 3.0]]]))
    out = ema.update(torch.tensor([[[3.0, 5.0]]]))
    assert torch.allclose(out, torch.tensor([[[2.0, 4.0]]]))

