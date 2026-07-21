import pandas as pd

from src.stock_news_impact.diagnostics import common_news_impact_diagnostics


def test_common_news_impact_detects_same_sign_common_move():
    gates = pd.DataFrame(
        [
            {
                "model": "S4",
                "config_id": "c",
                "event_id": "m1",
                "date": "2020-01-02",
                "hierarchy": "macro",
                "horizon": 1,
                "fold_id": 1,
                "seed": 42,
                "target_ticker": ticker,
                "gated_correction": 0.01,
                "utility": 0.001,
                "abnormal_volatility_response": 0.02,
            }
            for ticker in ["ADI", "AMD", "NVDA"]
        ]
    )
    detail, summary = common_news_impact_diagnostics(gates)
    assert detail["n_stocks"].iloc[0] == 3
    assert detail["same_sign_rate"].iloc[0] == 1.0
    assert summary["mean_commonality_ratio"].iloc[0] == 1.0
