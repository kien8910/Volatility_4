import pandas as pd

from src.stock_news_impact.loss_context_features import add_loss_context_features, loss_context_feature_columns


def _row(event_id, date, target_date, actual, stock_pred, news_corr):
    return {
        "event_id": event_id,
        "date": pd.Timestamp(date),
        "target_date": pd.Timestamp(target_date),
        "target_ticker": "ADI",
        "horizon": 1,
        "fold_id": 1,
        "seed": 42,
        "hierarchy": "macro",
        "event_scope": "macro",
        "actual_logvol": actual,
        "stock_prediction": stock_pred,
        "news_correction_proxy": news_corr,
        "abnormal_volatility_response": actual - stock_pred,
        "utility": 0.0,
    }


def test_loss_context_uses_only_realized_history_before_event_date():
    frame = pd.DataFrame(
        [
            _row("old", "2020-01-02", "2020-01-03", -3.0, -3.2, 0.1),
            _row("future", "2020-01-04", "2020-01-10", -2.0, -4.0, 1.0),
        ]
    )

    out = add_loss_context_features(frame)
    first = out.loc[out["event_id"].eq("old")].iloc[0]
    second = out.loc[out["event_id"].eq("future")].iloc[0]

    assert first["ticker_stock_loss_lag_mean_5"] == 0.0
    assert round(second["ticker_stock_loss_lag_mean_5"], 6) == 0.04
    # The second row's own large future loss must not be visible at its event date.
    assert second["ticker_stock_loss_lag_mean_5"] < 1.0


def test_loss_context_feature_columns_are_materialized():
    frame = pd.DataFrame([_row("e1", "2020-01-02", "2020-01-03", -3.0, -3.2, 0.1)])
    out = add_loss_context_features(frame)
    assert set(loss_context_feature_columns()).issubset(out.columns)
    assert {"event_common_utility", "event_common_utility_label"}.issubset(out.columns)


def test_event_history_context_excludes_same_day_news_outcomes():
    frame = pd.DataFrame(
        [
            _row("same_day_1", "2020-01-02", "2020-01-03", -3.0, -3.2, 0.1),
            _row("same_day_2", "2020-01-02", "2020-01-03", -3.0, -3.2, 0.1),
            _row("next_day", "2020-01-04", "2020-01-05", -3.0, -3.2, 0.1),
        ]
    )

    out = add_loss_context_features(frame)
    same_day = out.loc[out["event_id"].eq("same_day_2")].iloc[0]
    next_day = out.loc[out["event_id"].eq("next_day")].iloc[0]

    assert same_day["event_history_utility_mean_5"] == 0.0
    assert next_day["event_history_utility_mean_5"] > 0.0
