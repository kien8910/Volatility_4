from __future__ import annotations

import numpy as np
import pandas as pd


def oracle_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    stock_err = frame["actual_logvol"].astype(float) - frame["stock_prediction"].astype(float)
    adjusted = frame["stock_prediction"].astype(float) + frame["news_correction_proxy"].astype(float)
    news_err = frame["actual_logvol"].astype(float) - adjusted
    out = frame[["hierarchy", "target_ticker", "horizon", "fold_id", "seed"]].copy()
    out["oracle_gate"] = (np.square(news_err) < np.square(stock_err)).astype(int)
    out["oracle_utility"] = np.square(stock_err) - np.square(news_err)
    return (
        out.groupby(["hierarchy", "horizon"], as_index=False)
        .agg(positive_rate=("oracle_gate", "mean"), mean_oracle_utility=("oracle_utility", "mean"), n=("oracle_gate", "size"))
        .sort_values(["hierarchy", "horizon"])
    )
