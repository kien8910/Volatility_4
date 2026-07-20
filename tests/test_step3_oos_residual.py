import numpy as np
import pandas as pd
import pytest

from src.decomposition.residual_builder import SEMICONDUCTOR_TICKERS
from src.decomposition.walk_forward_predictor import build_state_residuals


def test_fixed_step3_tickers():
    assert SEMICONDUCTOR_TICKERS == ["ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MU", "NVDA", "QCOM", "TXN"]


def test_state_residual_is_actual_minus_prediction():
    pred = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "target_date": pd.to_datetime(["2020-01-02"]),
        "ticker": ["ADI"],
        "horizon": [1],
        "fold_id": [-1],
        "actual_target": [-4.0],
        "p_prediction": [-4.2],
        "residual_target": [0.2],
        "model_name": ["HAR-Ridge"],
        "is_oos": [1],
        "max_training_target_date": pd.to_datetime(["2019-12-31"]),
    })
    labels = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "base_split": ["train"], "analysis_split": ["train"]})
    state = build_state_residuals(pred, pd.DataFrame(), labels)
    assert state.loc[0, "residual_state_h1"] == pytest.approx(np.float64(state.loc[0, "actual_logvol_gk"] - state.loc[0, "p_prediction_h1"]))
    assert state.loc[0, "is_oos"] == 1
