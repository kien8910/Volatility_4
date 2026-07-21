import pandas as pd

from src.stock_news_impact.placebo_tests import wrong_ticker_placebo


def test_wrong_ticker_placebo_changes_ticker():
    pairs = pd.DataFrame([{"target_ticker": "ADI", "is_direct_target": 1}])
    out = wrong_ticker_placebo(pairs, ["ADI", "AMD"])
    assert out["target_ticker"].iloc[0] == "AMD"
    assert out["placebo_type"].iloc[0] == "wrong_ticker"

