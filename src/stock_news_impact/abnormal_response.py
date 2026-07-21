from __future__ import annotations

import numpy as np
import pandas as pd


def select_stock_only_predictions(step6_predictions: pd.DataFrame) -> pd.DataFrame:
    stock = step6_predictions.loc[step6_predictions["model"].astype(str).eq("stock_only")].copy()
    if stock.empty:
        raise ValueError("Step 7 requires stock_only rows in Step 6 predictions_validation.parquet.")
    return stock


def build_abnormal_response(stock_predictions: pd.DataFrame) -> pd.DataFrame:
    stock = select_stock_only_predictions(stock_predictions)
    out = stock[
        [
            "date",
            "target_date",
            "ticker",
            "horizon",
            "actual_logvol",
            "final_prediction",
            "split",
            "fold_id",
            "seed",
            "p_prediction",
            "stock_residual_prediction",
        ]
    ].copy()
    out = out.rename(columns={"final_prediction": "stock_prediction"})
    out["abnormal_volatility_response"] = out["actual_logvol"].astype(float) - out["stock_prediction"].astype(float)
    return out.sort_values(["date", "ticker", "horizon", "fold_id", "seed"]).reset_index(drop=True)


def stock_loss_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    err = out["actual_logvol"].astype(float) - out["stock_prediction"].astype(float)
    out["stock_squared_error"] = np.square(err)
    return out
