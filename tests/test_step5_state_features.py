import numpy as np
import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.graph.data_loader import Step4Inputs, validate_inputs
from src.graph.panel_builder import build_panels, build_sample_table
from src.regime_graph.state_features import StateFeatureScaler, build_state_feature_frame, state_feature_matrix


def _samples(n_dates=50):
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")
    state_rows, target_rows = [], []
    horizons = [1, 5, 10, 22]
    for t, date in enumerate(dates):
        split = "train" if t < 30 else "validation"
        for j, ticker in enumerate(SEMICONDUCTOR_TICKERS):
            raw = 0.2 + 0.01 * j + 0.001 * t
            state_rows.append({"date": date, "ticker": ticker, "actual_logvol_gk": raw, "residual_state_h1": raw - 0.1, "is_oos": 1, "base_split": split})
            for h in horizons:
                if t + h < n_dates:
                    actual = raw + 0.01
                    p = raw
                    target_rows.append({"date": date, "target_date": dates[t + h], "ticker": ticker, "horizon": h, "base_split": split, "fold_id": 0, "actual_target": actual, "p_prediction": p, "residual_target": actual - p, "is_oos": 1})
    inputs = validate_inputs(Step4Inputs(pd.DataFrame(state_rows), pd.DataFrame(target_rows), pd.DataFrame(target_rows), None, None), SEMICONDUCTOR_TICKERS, horizons)
    panels = build_panels(inputs.residual_state, inputs.residual_targets, SEMICONDUCTOR_TICKERS, horizons)
    return build_sample_table(panels, lookback=22)


def test_state_features_are_finite_and_origin_only():
    samples = _samples()
    frame = build_state_feature_frame(samples)
    matrix = state_feature_matrix(frame)
    assert matrix.shape[0] == len(samples.sample_dates)
    assert np.isfinite(matrix).all()
    first_window_last_mean = samples.raw_windows[0, :, -1].mean()
    assert frame.iloc[0]["market_logvol_mean"] == first_window_last_mean


def test_state_scaler_fit_train_only():
    train = np.ones((3, 2), dtype=np.float32)
    validation = np.full((1, 2), 100.0, dtype=np.float32)
    all_x = np.vstack([train, validation])
    scaler = StateFeatureScaler().fit(all_x[:3])
    transformed = scaler.transform(all_x)
    assert np.allclose(transformed[:3], 0.0)
    assert transformed[3].mean() > 90.0

