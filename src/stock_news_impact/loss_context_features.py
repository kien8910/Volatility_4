from __future__ import annotations

import numpy as np
import pandas as pd


ROLLING_WINDOWS = (5, 22)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.astype(float).replace(0.0, np.nan)
    return (num.astype(float) / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _prediction_level_losses(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "target_date", "target_ticker", "horizon", "fold_id", "seed"]
    keep = cols + ["actual_logvol", "stock_prediction", "news_correction_proxy", "abnormal_volatility_response"]
    base = frame[keep].drop_duplicates(cols).copy()
    stock_err = base["actual_logvol"].astype(float) - base["stock_prediction"].astype(float)
    news_pred = base["stock_prediction"].astype(float) + base["news_correction_proxy"].astype(float)
    news_err = base["actual_logvol"].astype(float) - news_pred
    base["ctx_stock_loss"] = np.square(stock_err)
    base["ctx_news_loss"] = np.square(news_err)
    base["ctx_utility"] = base["ctx_stock_loss"] - base["ctx_news_loss"]
    base["ctx_abs_abnormal"] = base["abnormal_volatility_response"].astype(float).abs()
    return base


def _rolling_prediction_history(base: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    value_cols = ["ctx_stock_loss", "ctx_news_loss", "ctx_utility", "ctx_abs_abnormal"]
    for _, grp in base.sort_values("target_date").groupby(group_cols, dropna=False):
        hist = grp[["target_date", *group_cols]].copy()
        for window in ROLLING_WINDOWS:
            rolled = grp[value_cols].rolling(window, min_periods=1).mean()
            for col in value_cols:
                hist[f"{prefix}_{col.replace('ctx_', '')}_lag_mean_{window}"] = rolled[col].to_numpy()
        rows.append(hist)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(columns={"target_date": "history_available_date"})


def _merge_history_asof(frame: pd.DataFrame, history: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    if history.empty:
        return frame
    out = []
    left_cols = set(frame.columns)
    for keys, left_grp in frame.sort_values("date").groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mask = pd.Series(True, index=history.index)
        for col, value in zip(by_cols, keys):
            mask &= history[col].eq(value)
        right_grp = history.loc[mask].sort_values("history_available_date")
        if right_grp.empty:
            merged = left_grp.copy()
        else:
            merged = pd.merge_asof(
                left_grp.sort_values("date"),
                right_grp,
                left_on="date",
                right_on="history_available_date",
                direction="backward",
                allow_exact_matches=False,
            )
            for col in by_cols:
                right_col = f"{col}_y"
                left_col = f"{col}_x"
                if left_col in merged.columns:
                    merged = merged.rename(columns={left_col: col})
                if right_col in merged.columns:
                    merged = merged.drop(columns=[right_col])
        out.append(merged[[c for c in merged.columns if c in left_cols or c not in left_cols]])
    return pd.concat(out, ignore_index=True) if out else frame


def _event_common_labels(frame: pd.DataFrame) -> pd.DataFrame:
    common_mask = frame["hierarchy"].astype(str).isin(["macro", "sector"])
    group_cols = ["event_id", "date", "hierarchy", "horizon", "fold_id", "seed"]
    common = frame.loc[common_mask].copy()
    if common.empty:
        out = frame.copy()
        out["event_common_utility"] = 0.0
        out["event_common_positive_rate"] = 0.0
        out["event_common_stock_loss"] = 0.0
        out["event_common_news_loss"] = 0.0
        out["event_common_utility_label"] = -1
        return out
    stock_err = common["actual_logvol"].astype(float) - common["stock_prediction"].astype(float)
    news_err = common["actual_logvol"].astype(float) - (
        common["stock_prediction"].astype(float) + common["news_correction_proxy"].astype(float)
    )
    common["event_stock_loss_raw"] = np.square(stock_err)
    common["event_news_loss_raw"] = np.square(news_err)
    common["event_utility_raw"] = common["event_stock_loss_raw"] - common["event_news_loss_raw"]
    summary = (
        common.groupby(group_cols, as_index=False)
        .agg(
            event_common_utility=("event_utility_raw", "mean"),
            event_common_positive_rate=("event_utility_raw", lambda x: float(np.mean(np.asarray(x) > 0.0))),
            event_common_stock_loss=("event_stock_loss_raw", "mean"),
            event_common_news_loss=("event_news_loss_raw", "mean"),
        )
    )
    summary["event_common_loss_delta"] = summary["event_common_stock_loss"] - summary["event_common_news_loss"]
    out = frame.merge(summary, on=group_cols, how="left")
    for col in [
        "event_common_utility",
        "event_common_positive_rate",
        "event_common_stock_loss",
        "event_common_news_loss",
        "event_common_loss_delta",
    ]:
        out[col] = out[col].fillna(0.0).astype(float)
    out["event_common_utility_label"] = np.where(
        ~out["hierarchy"].astype(str).isin(["macro", "sector"]),
        -1,
        np.where(out["event_common_utility"] > 0.0, 1, 0),
    )
    return out


def _event_history(frame: pd.DataFrame) -> pd.DataFrame:
    common = frame.loc[frame["hierarchy"].astype(str).isin(["macro", "sector"])].copy()
    if common.empty:
        return pd.DataFrame()
    group_cols = ["date", "hierarchy", "horizon", "fold_id", "seed"]
    event_level = (
        common[group_cols + ["event_id", "event_common_utility", "event_common_positive_rate", "event_common_loss_delta"]]
        .drop_duplicates(["event_id", *group_cols])
        .groupby(group_cols, as_index=False)
        .agg(
            event_common_utility=("event_common_utility", "mean"),
            event_common_positive_rate=("event_common_positive_rate", "mean"),
            event_common_loss_delta=("event_common_loss_delta", "mean"),
        )
    )
    rows = []
    hist_group_cols = ["hierarchy", "horizon", "fold_id", "seed"]
    for _, grp in event_level.sort_values("date").groupby(hist_group_cols, dropna=False):
        hist = grp[["date", *hist_group_cols]].copy()
        values = grp[["event_common_utility", "event_common_positive_rate", "event_common_loss_delta"]]
        for window in ROLLING_WINDOWS:
            rolled = values.rolling(window, min_periods=1).mean()
            hist[f"event_history_utility_mean_{window}"] = rolled["event_common_utility"].to_numpy()
            hist[f"event_history_positive_rate_mean_{window}"] = rolled["event_common_positive_rate"].to_numpy()
            hist[f"event_history_loss_delta_mean_{window}"] = rolled["event_common_loss_delta"].to_numpy()
        rows.append(hist)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(columns={"date": "event_history_date"})


def add_loss_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add non-leaky loss/utility context for event-time adaptive gating.

    Current-row losses are kept as labels/diagnostics only. Feature columns are
    built from histories whose realized target date is strictly before the event
    date, so validation rows do not see their own future outcome.
    """

    out = _event_common_labels(frame.copy())
    base = _prediction_level_losses(out)

    ticker_history = _rolling_prediction_history(base, ["target_ticker", "horizon", "fold_id", "seed"], "ticker")
    out = _merge_history_asof(out, ticker_history, ["target_ticker", "horizon", "fold_id", "seed"])

    market_base = (
        base.groupby(["date", "target_date", "horizon", "fold_id", "seed"], as_index=False)
        .agg(
            ctx_stock_loss=("ctx_stock_loss", "mean"),
            ctx_news_loss=("ctx_news_loss", "mean"),
            ctx_utility=("ctx_utility", "mean"),
            ctx_abs_abnormal=("ctx_abs_abnormal", "mean"),
        )
    )
    market_history = _rolling_prediction_history(market_base, ["horizon", "fold_id", "seed"], "market_error")
    out = _merge_history_asof(out, market_history, ["horizon", "fold_id", "seed"])

    event_history = _event_history(out)
    if not event_history.empty:
        merged = []
        by_cols = ["hierarchy", "horizon", "fold_id", "seed"]
        for keys, left_grp in out.sort_values("date").groupby(by_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            mask = pd.Series(True, index=event_history.index)
            for col, value in zip(by_cols, keys):
                mask &= event_history[col].eq(value)
            right_grp = event_history.loc[mask].sort_values("event_history_date")
            if right_grp.empty:
                merged.append(left_grp)
                continue
            joined = pd.merge_asof(
                left_grp.sort_values("date"),
                right_grp,
                left_on="date",
                right_on="event_history_date",
                direction="backward",
                allow_exact_matches=False,
            )
            for col in by_cols:
                if f"{col}_x" in joined.columns:
                    joined = joined.rename(columns={f"{col}_x": col})
                if f"{col}_y" in joined.columns:
                    joined = joined.drop(columns=[f"{col}_y"])
            merged.append(joined)
        out = pd.concat(merged, ignore_index=True)

    for col in loss_context_feature_columns():
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0).astype(float)
    return out


def loss_context_feature_columns() -> list[str]:
    cols: list[str] = []
    for prefix in ["ticker", "market_error"]:
        for metric in ["stock_loss", "news_loss", "utility", "abs_abnormal"]:
            for window in ROLLING_WINDOWS:
                cols.append(f"{prefix}_{metric}_lag_mean_{window}")
    for metric in ["utility", "positive_rate", "loss_delta"]:
        for window in ROLLING_WINDOWS:
            cols.append(f"event_history_{metric}_mean_{window}")
    return cols
