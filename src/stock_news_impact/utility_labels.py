from __future__ import annotations

import numpy as np
import pandas as pd


def add_utility_labels(frame: pd.DataFrame, train_mask: pd.Series, margin_quantile: float = 0.25) -> pd.DataFrame:
    out = frame.copy()
    stock_err = out["actual_logvol"].astype(float) - out["stock_prediction"].astype(float)
    adjusted = out["stock_prediction"].astype(float) + out["news_correction_proxy"].astype(float)
    news_err = out["actual_logvol"].astype(float) - adjusted
    out["utility"] = np.square(stock_err) - np.square(news_err)
    train_utility = out.loc[train_mask.astype(bool), "utility"].abs()
    margin = float(train_utility.quantile(float(margin_quantile))) if len(train_utility) else 0.0
    out["utility_margin"] = margin
    out["utility_label"] = np.where(out["utility"] > margin, 1, np.where(out["utility"] < -margin, 0, -1))
    return out
