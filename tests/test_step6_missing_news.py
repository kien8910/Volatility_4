import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.news.text_preprocessing import build_hierarchy_features_from_panel


def test_missing_news_days_are_kept_with_masks():
    panel = pd.DataFrame([{"date": "2020-01-02", "ticker": ticker} for ticker in SEMICONDUCTOR_TICKERS])
    features = build_hierarchy_features_from_panel(panel, SEMICONDUCTOR_TICKERS)
    assert len(features) == 11
    assert features["has_any_dynamic_news"].sum() == 0
    assert features["macro_missing_mask"].eq(1).all()

