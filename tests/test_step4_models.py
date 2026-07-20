import torch

from src.graph.models import StaticGraphForecastModel


def test_static_graph_model_output_shape_and_final_prediction_identity():
    model = StaticGraphForecastModel(
        num_nodes=11,
        lookback=22,
        num_horizons=4,
        temporal_kind="linear",
        temporal_cfg={"hidden_dim": 16},
        graph_type="identity",
        fixed_adjacency=torch.eye(11),
    )
    x = torch.randn(5, 11, 22)
    residual_pred = model(x)
    p = torch.randn(5, 11, 4)
    final = p + residual_pred
    assert residual_pred.shape == (5, 11, 4)
    assert torch.allclose(final, p + residual_pred)


def test_none_graph_model_has_no_cross_adjacency():
    model = StaticGraphForecastModel(
        num_nodes=11,
        lookback=22,
        num_horizons=4,
        temporal_kind="small_tcn",
        temporal_cfg={"hidden_dim": 16, "channels": [8, 8], "kernel_size": 3, "dropout": 0.0},
        graph_type="none",
    )
    assert model.adjacency() is None

