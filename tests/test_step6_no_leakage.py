import numpy as np
import pandas as pd

from src.news.evaluator import flatten_step6_predictions


class Samples:
    sample_dates = pd.DatetimeIndex(["2020-01-02"])
    target_dates = np.array([["2020-01-03"]], dtype="datetime64[ns]")
    tickers = ["ADI"]
    horizons = [1]


def test_flatten_predictions_target_date_after_origin():
    raw = {
        "sample_index": np.array([0]),
        "actual_logvol": np.array([[[1.0]]]),
        "p_prediction": np.array([[[0.8]]]),
        "residual_actual": np.array([[[0.2]]]),
        "stock_residual_prediction": np.array([[[0.1]]]),
        "news_residual_correction": np.array([[[0.01]]]),
        "final_residual_prediction": np.array([[[0.11]]]),
        "final_prediction": np.array([[[0.91]]]),
        "qlike_loss": np.array([[[0.1]]]),
    }
    coverage = pd.DataFrame([{"date": "2020-01-02", "ticker": "ADI", "has_macro": 0, "has_sector": 0, "has_target_company": 0, "has_related_company": 0, "has_filing": 0, "has_any_dynamic_news": 0}])
    pred = flatten_step6_predictions(raw, Samples, coverage, "validation", 1, 42, {"model": "stock_only", "config_id": "c"}, 0.9)
    assert (pred["target_date"] > pred["date"]).all()

