import torch

from src.graph.checkpointing import load_checkpoint, save_checkpoint
from src.graph.models import StaticGraphForecastModel


def test_checkpoint_save_load_roundtrip(tmp_path):
    model = StaticGraphForecastModel(
        num_nodes=3,
        lookback=4,
        num_horizons=2,
        temporal_kind="linear",
        temporal_cfg={"hidden_dim": 8},
        graph_type="identity",
        fixed_adjacency=torch.eye(3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        epoch=3,
        best_metric=0.5,
        hyperparameters={"model_id": "G1"},
        seed=42,
        ticker_order=["A", "B", "C"],
        scaler_state={"mean": [[[0.0], [0.0], [0.0]]], "scale": [[[1.0], [1.0], [1.0]]]},
        adjacency=torch.eye(3),
    )
    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["epoch"] == 3
    assert loaded["best_validation_metric"] == 0.5
    assert loaded["ticker_order"] == ["A", "B", "C"]

