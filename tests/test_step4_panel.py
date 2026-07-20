import numpy as np
import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.graph.data_loader import Step4Inputs, validate_inputs
from src.graph.panel_builder import build_panels, build_sample_table


def _synthetic_inputs(n_dates=80):
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    state_rows = []
    target_rows = []
    horizons = [1, 5, 10, 22]
    for t_idx, date in enumerate(dates):
        split = "train" if t_idx < 30 else "validation" if t_idx < 60 else "test"
        for j, ticker in enumerate(SEMICONDUCTOR_TICKERS):
            raw = 0.1 + 0.01 * j + 0.001 * t_idx
            pred_h1 = raw - 0.01
            state_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "actual_logvol_gk": raw,
                    "p_prediction_h1": pred_h1,
                    "residual_state_h1": raw - pred_h1,
                    "is_oos": 1,
                    "base_split": split,
                }
            )
            for h in horizons:
                if t_idx + h < n_dates:
                    actual = raw + 0.001 * h
                    p = raw
                    target_rows.append(
                        {
                            "date": date,
                            "target_date": dates[t_idx + h],
                            "ticker": ticker,
                            "horizon": h,
                            "base_split": split,
                            "fold_id": 0,
                            "actual_target": actual,
                            "p_prediction": p,
                            "residual_target": actual - p,
                            "is_oos": 1,
                        }
                    )
    state = pd.DataFrame(state_rows)
    targets = pd.DataFrame(target_rows)
    return Step4Inputs(state, targets, targets.copy(), None, None)


def test_validate_inputs_and_build_complete_samples():
    inputs = validate_inputs(_synthetic_inputs(), SEMICONDUCTOR_TICKERS, [1, 5, 10, 22])
    panels = build_panels(inputs.residual_state, inputs.residual_targets, SEMICONDUCTOR_TICKERS, [1, 5, 10, 22])
    samples = build_sample_table(panels, lookback=22)
    assert panels.residual_state.shape[1] == 11
    assert samples.residual_windows.shape[1:] == (11, 22)
    assert np.isfinite(samples.target_residual).all()
    assert list(samples.tickers) == SEMICONDUCTOR_TICKERS

