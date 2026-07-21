import torch

from src.stock_news_impact.trainer import Step7RunConfig, build_model


def test_s5_utility_gate_accepts_utility_context_features():
    columns = {
        "event": ["e1", "e2"],
        "relation": ["r1"],
        "stock": ["s1"],
        "market": ["m1"],
        "utility_context": ["u1", "u2", "u3"],
    }
    cfg = {
        "target": {"horizons": [1, 5]},
        "gate": {"reliability_hidden_dim": 4, "horizon_embedding_dim": 2, "dropout": 0.0},
    }
    run_cfg = Step7RunConfig(model="S5_UtilityFactorizedGate", initial_probability=0.05)
    model = build_model(run_cfg, columns, cfg)
    out = model(
        torch.randn(3, 2),
        torch.randn(3, 1),
        torch.randn(3, 1),
        torch.randn(3, 4),
        torch.tensor([1, 5, 1]),
    )
    assert out["final_gate"].shape == (3,)
    assert torch.all(out["final_gate"] >= 0)
    assert torch.all(out["final_gate"] <= 1)
