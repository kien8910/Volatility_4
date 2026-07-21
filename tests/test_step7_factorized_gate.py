import torch

from src.stock_news_impact.factorized_gate import FactorizedGateModel


def test_factorized_gate_range_and_product():
    model = FactorizedGateModel(3, 2, 2, 1, horizons=[1, 5], initial_probability=0.10)
    out = model(torch.randn(4, 3), torch.randn(4, 2), torch.randn(4, 2), torch.randn(4, 1), torch.tensor([1, 5, 1, 5]))
    assert torch.all(out["final_gate"] >= 0)
    assert torch.all(out["final_gate"] <= 1)
    assert torch.allclose(out["final_gate"], out["reliability_gate"] * out["stock_relevance_gate"] * out["horizon_gate"])

