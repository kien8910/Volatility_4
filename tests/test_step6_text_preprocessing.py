import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.news.text_preprocessing import build_hierarchy_features_from_panel, normalize_text, stable_text_hash


def test_normalize_missing_text_values():
    for value in [None, float("nan"), "", "   ", "null", "None"]:
        text, missing = normalize_text(value)
        assert text == ""
        assert missing == 1


def test_build_hierarchy_features_preserves_11_tickers():
    rows = []
    for ticker in SEMICONDUCTOR_TICKERS:
        rows.append({"date": "2020-01-02", "ticker": ticker, "macro_category1": "Fed text", "sector_category1": "Chip text"})
    frame = build_hierarchy_features_from_panel(pd.DataFrame(rows), SEMICONDUCTOR_TICKERS)
    assert sorted(frame["ticker"].unique()) == sorted(SEMICONDUCTOR_TICKERS)
    assert frame["macro_text"].str.contains("[CATEGORY_1]", regex=False).all()
    assert frame["macro_text_hash"].nunique() == 1
    assert frame["macro_text_hash"].iloc[0] == stable_text_hash(frame["macro_text"].iloc[0])

