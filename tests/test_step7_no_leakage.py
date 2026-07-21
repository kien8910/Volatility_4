import pandas as pd

from src.stock_news_impact.novelty_features import add_novelty_features


def test_novelty_uses_previous_same_text_only():
    events = pd.DataFrame(
        [
            {"event_id": "e1", "date": "2020-01-02", "text_hash": "h"},
            {"event_id": "e2", "date": "2020-01-05", "text_hash": "h"},
        ]
    )
    out = add_novelty_features(events).sort_values("date")
    assert out["duplicate_frequency_to_date"].tolist() == [0.0, 1.0]
    assert out["days_since_same_text"].tolist()[1] == 3.0

