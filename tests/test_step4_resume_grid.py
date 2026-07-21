import pandas as pd

from src.graph.run_step4 import (
    append_edges_incremental,
    append_predictions_incremental,
    apply_quick_grid,
    completed_run_keys,
    enumerate_configs,
)


def _base_cfg():
    return {
        "experiment": {"seeds": [42, 123, 2026]},
        "window": {"lookbacks": [22, 44]},
        "temporal_encoder": {"options": ["linear", "small_tcn"]},
        "training": {"loss_options": ["mse", "huber"]},
        "graph": {"top_k": [2, 3], "embedding_dims": [8, 16], "directed_options": [True, False]},
        "search": {"include_models": ["G0", "G1", "G2", "G5"], "max_configs": None},
    }


def test_include_models_and_max_configs_filter_grid():
    cfg = _base_cfg()
    cfg["search"]["include_models"] = ["G1", "G5"]
    cfg["search"]["max_configs"] = 5
    configs = enumerate_configs(cfg)
    assert len(configs) == 5
    assert {item.model for item in configs}.issubset({"G1", "G5"})


def test_quick_grid_shrinks_grid_to_pilot_defaults():
    cfg = apply_quick_grid(_base_cfg())
    assert cfg["search"]["include_models"] == ["G0", "G1", "G2", "G5"]
    assert cfg["experiment"]["seeds"] == [42, 123]
    assert cfg["window"]["lookbacks"] == [22]
    assert cfg["training"]["loss_options"] == ["mse"]
    assert cfg["graph"]["directed_options"] == [False]


def test_incremental_predictions_are_deduplicated_and_resume_keys_are_available(tmp_path):
    row = {
        "config_id": "G1__L22__linear__mse__identity",
        "split": "validation",
        "fold_id": 1,
        "seed": 42,
        "model": "G1",
        "date": pd.Timestamp("2020-01-01"),
        "target_date": pd.Timestamp("2020-01-02"),
        "ticker": "ADI",
        "horizon": 1,
        "qlike_loss": 0.1,
    }
    pred = pd.DataFrame([row])
    append_predictions_incremental(pred, tmp_path)
    append_predictions_incremental(pred.assign(qlike_loss=0.2), tmp_path)
    saved = pd.read_parquet(tmp_path / "predictions_validation.parquet")
    assert len(saved) == 1
    assert float(saved.iloc[0]["qlike_loss"]) == 0.2
    assert completed_run_keys(tmp_path) == {("G1__L22__linear__mse__identity", 1, 42)}


def test_incremental_edges_are_deduplicated(tmp_path):
    edge = pd.DataFrame(
        [
            {
                "config_id": "G2__L22__linear__mse__correlation__k2",
                "model": "G2",
                "fold_id": 1,
                "seed": 42,
                "source": "ADI",
                "target": "AMD",
                "weight": 0.5,
            }
        ]
    )
    append_edges_incremental(edge, tmp_path)
    append_edges_incremental(edge.assign(weight=0.7), tmp_path)
    saved = pd.read_csv(tmp_path / "graph_edges.csv")
    assert len(saved) == 1
    assert float(saved.iloc[0]["weight"]) == 0.7
