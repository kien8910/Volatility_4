from __future__ import annotations

import pandas as pd

from src.stock_news_impact.abnormal_response import build_abnormal_response, stock_loss_columns
from src.stock_news_impact.event_features import add_event_features, event_feature_columns
from src.stock_news_impact.frozen_branches import news_correction_proxy
from src.stock_news_impact.novelty_features import add_novelty_features, novelty_feature_columns
from src.stock_news_impact.relation_features import add_relation_features, relation_feature_columns
from src.stock_news_impact.stock_features import add_stock_features, market_context_features, market_feature_columns, stock_feature_columns
from src.stock_news_impact.utility_labels import add_utility_labels


def build_gate_feature_frame(events: pd.DataFrame, pairs: pd.DataFrame, step6_predictions: pd.DataFrame, selected_branch: dict, cfg: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    abnormal = build_abnormal_response(step6_predictions)
    abnormal = stock_loss_columns(abnormal)
    correction = news_correction_proxy(step6_predictions, selected_branch)
    events_features = add_novelty_features(add_event_features(events))
    pairs_features = add_relation_features(pairs)
    frame = pairs_features.merge(events_features, on=["event_id", "date", "hierarchy", "source_ticker", "context_ticker", "event_scope"], how="left")
    frame = frame.merge(
        abnormal,
        left_on=["date", "target_date", "target_ticker", "horizon", "fold_id", "seed"],
        right_on=["date", "target_date", "ticker", "horizon", "fold_id", "seed"],
        how="inner",
    )
    if "ticker" in frame.columns:
        frame = frame.drop(columns=["ticker"])
    if "split_x" in frame.columns:
        frame = frame.rename(columns={"split_x": "split"})
    if "split_y" in frame.columns:
        frame = frame.drop(columns=["split_y"])
    frame = frame.merge(
        correction,
        left_on=["date", "target_date", "target_ticker", "horizon", "fold_id", "seed"],
        right_on=["date", "target_date", "ticker", "horizon", "fold_id", "seed"],
        how="left",
        suffixes=("", "_corr"),
    )
    # The correction proxy is joined through the Step 6 ticker column. After the
    # join, target_ticker remains the authoritative event-stock target. Drop the
    # right-side ticker before temporarily renaming target_ticker for stock feature
    # construction; otherwise pandas creates duplicate "ticker" columns and
    # out["ticker"] becomes a DataFrame instead of a Series.
    for col in ["ticker", "ticker_corr"]:
        if col in frame.columns:
            frame = frame.drop(columns=[col])
    frame["news_correction_proxy"] = frame["news_correction_proxy"].fillna(0.0)
    frame = add_stock_features(frame.rename(columns={"target_ticker": "ticker"})).rename(columns={"ticker": "target_ticker"})
    market = market_context_features(abnormal.rename(columns={"ticker": "target_ticker"}))
    frame = frame.merge(market, on=["date", "fold_id", "seed", "horizon"], how="left")
    train_mask = frame["split"].astype(str).eq("validation") & frame["fold_id"].astype(int).ne(frame["fold_id"].astype(int).max())
    frame = add_utility_labels(frame, train_mask, float(cfg["utility_supervision"]["margin_quantiles"][0]))
    cols = {
        "event": event_feature_columns() + novelty_feature_columns(),
        "relation": relation_feature_columns(),
        "stock": stock_feature_columns(),
        "market": market_feature_columns(),
    }
    for values in cols.values():
        for col in values:
            frame[col] = frame[col].fillna(0.0).astype(float)
    return frame.reset_index(drop=True), cols
