import numpy as np
import pandas as pd
import pytest

from src.baselines.data import SEMICONDUCTOR_TICKERS, add_time_series_features, recompute_gk, split_dates_for_fold
from src.baselines.metrics import qlike
from src.baselines.models import dlinear_history


def toy_panel(n=90):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for ticker in ["ADI", "AMAT"]:
        close = 100 + np.arange(n) * 0.1
        rows.append(pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.99,
            "close": close * 1.01,
            "ohlc_valid": True,
            "log_return": np.r_[np.nan, np.diff(np.log(close * 1.01))],
        }))
    df = pd.concat(rows, ignore_index=True)
    checked = recompute_gk(df, 1.0e-12)
    df["gk_variance_raw"] = checked["gk_variance_raw_recomputed"]
    df["gk_variance"] = checked["gk_variance_recomputed"]
    df["gk_nonpositive_flag"] = df["gk_variance_raw"] <= 0
    df["logvol_gk"] = checked["logvol_gk_recomputed"]
    for h in [1, 5, 10, 22]:
        df[f"target_date_h{h}"] = df.groupby("ticker")["date"].shift(-h)
        df[f"target_logvol_gk_h{h}"] = df.groupby("ticker")["logvol_gk"].shift(-h)
    return df


def test_fixed_semiconductor_tickers():
    assert SEMICONDUCTOR_TICKERS == ["ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MU", "NVDA", "QCOM", "TXN"]
    assert len(SEMICONDUCTOR_TICKERS) == 11


def test_gk_and_logvol_formula():
    df = toy_panel(3).head(1)
    row = df.iloc[0]
    expected_raw = 0.5 * np.log(row.high / row.low) ** 2 - (2 * np.log(2) - 1) * np.log(row.close / row.open) ** 2
    assert row.gk_variance_raw == pytest.approx(expected_raw)
    assert row.logvol_gk == pytest.approx(0.5 * np.log(max(expected_raw, 1.0e-12)))


def test_target_shift_and_target_date_order():
    df = toy_panel(30)
    part = df[df.ticker == "ADI"].reset_index(drop=True)
    assert part.loc[0, "target_logvol_gk_h5"] == pytest.approx(part.loc[5, "logvol_gk"])
    assert part.loc[0, "target_date_h5"] > part.loc[0, "date"]


def test_har_features_use_only_present_and_past():
    df = toy_panel(30)
    feat = add_time_series_features(df, [22], [1])
    part = feat[feat.ticker == "ADI"].reset_index(drop=True)
    idx = 21
    assert part.loc[idx, "har_w"] == pytest.approx(part.loc[idx - 4:idx, "logvol_gk"].mean())
    assert part.loc[idx, "har_m"] == pytest.approx(part.loc[0:idx, "logvol_gk"].mean())
    assert part.loc[idx, "lag_0"] == pytest.approx(part.loc[idx, "logvol_gk"])
    assert part.loc[idx, "lag_21"] == pytest.approx(part.loc[0, "logvol_gk"])


def test_qlike_formula_and_garch_logvol_conversion():
    y_true = np.array([0.5 * np.log(4.0)])
    y_pred = np.array([0.5 * np.log(2.0)])
    loss, clipped = qlike(y_true, y_pred, 1.0e-12)
    ratio = 4.0 / 2.0
    assert loss[0] == pytest.approx(ratio - np.log(ratio) - 1.0)
    assert not clipped[0]
    garch_var = 2.0
    assert y_pred[0] == pytest.approx(0.5 * np.log(max(garch_var, 1.0e-12)))


def test_split_no_overlap():
    folds = pd.DataFrame({
        "fold_id": [1, 1, 1, 1],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"]),
        "role": ["train", "train", "validation", "validation"],
    })
    train, val = split_dates_for_fold(folds, 1)
    assert not train & val


def test_dlinear_reproducible_same_seed():
    df = add_time_series_features(toy_panel(90), [22], [1])
    target = "target_logvol_gk_h1"
    train = df[(df.ticker == "ADI") & (df.date < "2020-03-15") & df.valid_h1]
    val = df[(df.ticker == "ADI") & (df.date >= "2020-03-15") & df.valid_h1].head(5)
    a = dlinear_history(train, val, target, lookback=22, seed=7, max_epochs=5, patience=3, learning_rate=0.01)
    b = dlinear_history(train, val, target, lookback=22, seed=7, max_epochs=5, patience=3, learning_rate=0.01)
    np.testing.assert_allclose(a.y_pred, b.y_pred, rtol=0, atol=1.0e-7)
