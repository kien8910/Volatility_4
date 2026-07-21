import pandas as pd

from src.stock_news_impact.event_stock_pairs import build_event_stock_pairs, validate_pairs


def test_target_event_pairs_only_context_ticker():
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "date": pd.Timestamp("2020-01-02"),
                "source_ticker": "ADI",
                "context_ticker": "ADI",
                "hierarchy": "target_company",
                "event_scope": "firm_specific",
            }
        ]
    )
    pred = pd.DataFrame(
        [
            {"date": "2020-01-02", "target_date": "2020-01-03", "ticker": t, "horizon": 1, "split": "validation", "fold_id": 1, "seed": 42}
            for t in ["ADI", "AMD"]
        ]
    )
    pairs = build_event_stock_pairs(events, pred, ["ADI", "AMD"], "target_only")
    validate_pairs(pairs)
    assert pairs["target_ticker"].tolist() == ["ADI"]
    assert pairs["is_direct_target"].eq(1).all()

