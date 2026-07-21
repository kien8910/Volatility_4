from __future__ import annotations

import numpy as np
import pandas as pd


def add_stock_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ticker" not in out.columns:
        raise ValueError("add_stock_features requires a ticker column.")
    if out.columns.tolist().count("ticker") > 1:
        raise ValueError("add_stock_features received duplicate ticker columns.")
    out["stock_prediction"] = out["stock_prediction"].astype(float)
    out["stock_residual_prediction"] = out["stock_residual_prediction"].astype(float)
    out["abs_stock_residual_prediction"] = out["stock_residual_prediction"].abs()
    out["ticker_code"] = out["ticker"].astype("category").cat.codes.astype(float)
    out["ticker_code"] = out["ticker_code"] / max(float(out["ticker_code"].max()), 1.0)
    out["horizon_scaled"] = out["horizon"].astype(float) / max(float(out["horizon"].max()), 1.0)
    return out


def stock_feature_columns() -> list[str]:
    return ["stock_prediction", "stock_residual_prediction", "abs_stock_residual_prediction", "ticker_code", "horizon_scaled"]


def market_context_features(stock_predictions: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "fold_id", "seed", "horizon"]
    grouped = stock_predictions.groupby(cols, as_index=False).agg(
        market_mean_prediction=("stock_prediction", "mean"),
        market_prediction_dispersion=("stock_prediction", "std"),
        market_mean_abs_residual_pred=("stock_residual_prediction", lambda x: float(np.mean(np.abs(x)))),
    )
    grouped["market_prediction_dispersion"] = grouped["market_prediction_dispersion"].fillna(0.0)
    return grouped


def market_feature_columns() -> list[str]:
    return ["market_mean_prediction", "market_prediction_dispersion", "market_mean_abs_residual_pred"]
