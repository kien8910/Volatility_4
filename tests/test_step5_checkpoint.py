import torch

from src.regime_graph.checkpointing import load_step5_checkpoint, save_step5_checkpoint
from src.regime_graph.regime_graph_model import RegimeGraphForecastModel


def test_step5_checkpoint_roundtrip(tmp_path):
    model = RegimeGraphForecastModel(
        num_nodes=3,
        lookback=4,
        num_horizons=2,
        temporal_cfg={"hidden_dim": 8, "channels": [4, 4], "kernel_size": 3, "dropout": 0.0, "activation": "relu"},
        graph_embedding_dim=2,
        top_k=1,
        directed=False,
        num_graphs=1,
        model_id="S5-B0",
        state_dim=7,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "best.pt"
    save_step5_checkpoint(path, model, opt, None, 2, 0.3, {"config_id": "x"}, 42, ["A", "B", "C"], {"mean": [0], "scale": [1]}, {"mean": [0], "scale": [1]})
    loaded = load_step5_checkpoint(path, model, opt)
    assert loaded["epoch"] == 2
    assert loaded["best_validation_qlike"] == 0.3

