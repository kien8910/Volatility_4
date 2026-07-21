import pandas as pd

from src.stock_news_impact.run_step7 import feature_columns_compatible


def test_s5_requires_utility_context_columns_in_feature_schema():
    frame = pd.DataFrame(
        {
            "event_a": [1.0],
            "relation_a": [1.0],
            "stock_a": [1.0],
            "market_a": [1.0],
            "ticker_utility_lag_mean_5": [0.0],
        }
    )
    cfg = {"search": {"include_models": ["S0_StockOnly_G5", "S5_UtilityFactorizedGate"]}}
    stale_columns = {
        "event": ["event_a"],
        "relation": ["relation_a"],
        "stock": ["stock_a"],
        "market": ["market_a"],
    }
    fresh_columns = {
        **stale_columns,
        "utility_context": ["ticker_utility_lag_mean_5"],
    }

    assert not feature_columns_compatible(frame, stale_columns, cfg)
    assert feature_columns_compatible(frame, fresh_columns, cfg)


def test_legacy_gate_schema_can_skip_utility_context_when_s5_not_requested():
    frame = pd.DataFrame({"event_a": [1.0], "relation_a": [1.0], "stock_a": [1.0], "market_a": [1.0]})
    cfg = {"search": {"include_models": ["S0_StockOnly_G5", "S2_FixedSmallGate"]}}
    columns = {
        "event": ["event_a"],
        "relation": ["relation_a"],
        "stock": ["stock_a"],
        "market": ["market_a"],
    }

    assert feature_columns_compatible(frame, columns, cfg)
