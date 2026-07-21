import torch

from src.regime_graph.regime_gate import RegimeGate
from src.regime_graph.regime_graph_model import RegimeGraphForecastModel


def test_regime_gate_weights_sum_to_one():
    gate = RegimeGate(state_dim=7, hidden_dim=8, num_regimes=3, temperature=1.0)
    weights = gate(torch.randn(5, 7))
    assert weights.shape == (5, 3)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-6)


def test_regime_model_output_shape():
    model = RegimeGraphForecastModel(
        num_nodes=11,
        lookback=22,
        num_horizons=4,
        temporal_cfg={"hidden_dim": 16, "channels": [8, 8], "kernel_size": 3, "dropout": 0.0, "activation": "relu"},
        graph_embedding_dim=8,
        top_k=3,
        directed=False,
        num_graphs=2,
        model_id="S5-R",
        state_dim=7,
    )
    pred, aux = model(torch.randn(6, 11, 22), torch.randn(6, 7))
    assert pred.shape == (6, 11, 4)
    assert aux["regime_weights"].shape == (6, 2)

