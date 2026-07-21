import pandas as pd

from src.regime_graph.diagnostics import regime_usage_from_predictions


def test_regime_usage_detects_collapse():
    rows = []
    for i in range(10):
        rows.append({"config_id": "x", "model": "S5-R", "split": "validation", "fold_id": 1, "seed": 42, "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "regime_argmax": 1, "regime_weight_1": 0.96, "regime_weight_2": 0.04})
    usage = regime_usage_from_predictions(pd.DataFrame(rows))
    assert bool(usage.iloc[0]["collapse_90"])
    assert usage.iloc[0]["effective_regimes"] < 2

