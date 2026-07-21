import pandas as pd

from src.stock_news_impact.event_stock_pairs import validate_pairs


def test_step7_rejects_same_day_target():
    pairs = pd.DataFrame(
        [{"event_id": "e", "date": "2020-01-02", "target_date": "2020-01-02", "target_ticker": "ADI", "horizon": 0, "fold_id": 1, "seed": 42, "is_direct_target": 1}]
    )
    try:
        validate_pairs(pairs)
    except ValueError as exc:
        assert "target_date" in str(exc)
    else:
        raise AssertionError("same-day target should fail")

