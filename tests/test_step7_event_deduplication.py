import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.news.text_preprocessing import build_hierarchy_features_from_panel
from src.stock_news_impact.event_builder import build_news_events


def test_filing_dedupes_by_ticker_and_text_hash():
    rows = []
    for date in ["2020-01-02", "2020-01-03"]:
        for ticker in SEMICONDUCTOR_TICKERS:
            rows.append({"date": date, "ticker": ticker, "filing_financialStatement": "same filing"})
    features = build_hierarchy_features_from_panel(pd.DataFrame(rows), SEMICONDUCTOR_TICKERS)
    cfg = {"events": {"include_hierarchies": ["filing"], "treat_filing_as_context": True}}
    events = build_news_events(features, cfg)
    assert len(events) == 11
    assert events["is_dynamic_news"].eq(0).all()

