from __future__ import annotations

import pandas as pd


def add_novelty_features(events: pd.DataFrame) -> pd.DataFrame:
    out = events.sort_values(["text_hash", "date"]).copy()
    out["duplicate_frequency_to_date"] = out.groupby("text_hash").cumcount().astype(float)
    previous = out.groupby("text_hash")["date"].shift(1)
    out["days_since_same_text"] = (pd.to_datetime(out["date"]) - pd.to_datetime(previous)).dt.days.fillna(999).astype(float)
    out["news_novelty"] = (out["duplicate_frequency_to_date"].eq(0)).astype(float)
    return out.sort_values("event_id").reset_index(drop=True)


def novelty_feature_columns() -> list[str]:
    return ["duplicate_frequency_to_date", "days_since_same_text", "news_novelty"]
