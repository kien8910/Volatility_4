import numpy as np
import pytest

from src.decomposition.residual_diagnostics import diagnostics_by_ticker_split
from src.decomposition.spectral_diagnostics import acf_energy, low_frequency_ratio


def test_acf_energy_known_ar_like_series():
    x = np.arange(30, dtype=float)
    assert acf_energy(x, 3) > 0


def test_lfr_bounds():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    lfr = low_frequency_ratio(x, 0.10)
    assert 0 <= lfr <= 1


def test_variance_ratio_formula():
    import pandas as pd

    raw = np.array([1, 2, 3, 4, 5], dtype=float)
    res = raw * 0.5
    df = pd.DataFrame({"ticker": "ADI", "base_split": "train", "date": pd.bdate_range("2020-01-01", periods=5), "actual_logvol_gk": raw, "residual_state_h1": res})
    out = diagnostics_by_ticker_split(df, 2, 3, 0.10, 1e-12)
    assert out.loc[0, "variance_ratio"] == pytest.approx(np.var(res, ddof=1) / np.var(raw, ddof=1))
