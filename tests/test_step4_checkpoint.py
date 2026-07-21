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


def test_checkpoint_load_moves_model_and_optimizer_state_to_requested_device(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticGraphForecastModel(
        num_nodes=3,
        lookback=4,
        num_horizons=2,
        temporal_kind="linear",
        temporal_cfg={"hidden_dim": 8},
        graph_type="identity",
        fixed_adjacency=torch.eye(3),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 4, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        epoch=1,
        best_metric=0.4,
        hyperparameters={"model_id": "G1"},
        seed=42,
        ticker_order=["A", "B", "C"],
        scaler_state={"mean": [[[0.0], [0.0], [0.0]]], "scale": [[[1.0], [1.0], [1.0]]]},
        adjacency=torch.eye(3, device=device),
    )
    fresh = StaticGraphForecastModel(
        num_nodes=3,
        lookback=4,
        num_horizons=2,
        temporal_kind="linear",
        temporal_cfg={"hidden_dim": 8},
        graph_type="identity",
        fixed_adjacency=torch.eye(3),
    )
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    load_checkpoint(path, fresh, fresh_opt, map_location=device)
    assert next(fresh.parameters()).device == device
    for state in fresh_opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device == device
