import numpy as np
import pandas as pd

from src.decomposition.walk_forward_predictor import walk_forward_predictions


def toy_panel(n=80):
    dates = pd.bdate_range("2020-01-01", periods=n)
    v = np.sin(np.arange(n) / 10) * 0.1 - 4
    df = pd.DataFrame({"date": dates, "ticker": "ADI", "logvol_gk": v, "log_return": 0.01, "ohlc_valid": True})
    df["har_d"] = df["logvol_gk"]
    df["har_w"] = df["logvol_gk"].rolling(5, min_periods=5).mean()
    df["har_m"] = df["logvol_gk"].rolling(22, min_periods=22).mean()
    for h in [1]:
        df[f"target_date_h{h}"] = df["date"].shift(-h)
        df[f"target_logvol_gk_h{h}"] = df["logvol_gk"].shift(-h)
        df[f"valid_origin_h{h}"] = df["har_m"].notna() & df[f"target_logvol_gk_h{h}"].notna()
    return df


def test_walk_forward_uses_only_available_targets():
    panel = toy_panel()
    split = pd.DataFrame({"date": panel.date, "is_locked_test": 0, "base_split": "development"})
    folds = pd.DataFrame({"fold_id": 1, "date": panel.date.iloc[50:60], "role": "validation"})
    p_cfg = {"model_name": "HAR-Ridge", "selected": [{"ticker": "ADI", "horizon": 1, "hyperparams": "{\"alpha\": 1.0}"}]}
    pred = walk_forward_predictions(panel, split, folds, p_cfg, [1], initial_training_days=30)
    assert len(pred) > 0
    assert pred["max_training_target_date"].le(pred["date"]).all()
    assert pred["target_date"].gt(pred["date"]).all()
    assert pred["is_oos"].eq(1).all()
