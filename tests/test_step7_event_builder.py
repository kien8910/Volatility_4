import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.news.text_preprocessing import build_hierarchy_features_from_panel
from src.stock_news_impact.event_builder import build_news_events, validate_events


def test_step7_event_builder_macro_dedup_and_target_events():
    panel = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "ticker": ticker,
                "macro_category1": "same macro",
                "sector_category1": "same sector",
                "targetCompany_category1": f"{ticker} target",
            }
            for ticker in SEMICONDUCTOR_TICKERS
        ]
    )
    features = build_hierarchy_features_from_panel(panel, SEMICONDUCTOR_TICKERS)
    cfg = {"events": {"include_hierarchies": ["macro", "sector", "target_company"], "treat_filing_as_context": True}}
    events = build_news_events(features, cfg)
    validate_events(events)
    assert len(events[events.hierarchy.eq("macro")]) == 1
    assert len(events[events.hierarchy.eq("sector")]) == 1
    assert len(events[events.hierarchy.eq("target_company")]) == 11

