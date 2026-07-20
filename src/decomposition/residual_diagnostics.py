from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from .spectral_diagnostics import acf_energy, acf_values, low_frequency_ratio


def diagnostics_by_ticker_split(state: pd.DataFrame, acf_lag: int, acf_max_lag: int, low_fraction: float, epsilon: float) -> pd.DataFrame:
    rows = []
    for (ticker, split), part in state.groupby(["ticker", "base_split"], dropna=False):
        part = part.sort_values("date")
        raw = part["actual_logvol_gk"].to_numpy(dtype=float)
        residual = part["residual_state_h1"].to_numpy(dtype=float)
        raw_e = acf_energy(raw, acf_lag)
        res_e = acf_energy(residual, acf_lag)
        lfr_raw = low_frequency_ratio(raw, low_fraction, epsilon)
        lfr_res = low_frequency_ratio(residual, low_fraction, epsilon)
        vr = float(np.nanvar(residual, ddof=1) / (np.nanvar(raw, ddof=1) + epsilon)) if len(part) > 2 else np.nan
        lb = {}
        for lag in [5, 10, 22]:
            try:
                lb[f"ljungbox_p_lag{lag}"] = float(acorr_ljungbox(residual[np.isfinite(residual)], lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
            except Exception:
                lb[f"ljungbox_p_lag{lag}"] = np.nan
        acf66 = acf_values(residual, acf_max_lag)
        rows.append({
            "ticker": ticker,
            "base_split": split,
            "n": int(len(part)),
            "acf_energy_raw": raw_e,
            "acf_energy_residual": res_e,
            "acf_energy_reduction": float(1.0 - res_e / (raw_e + epsilon)),
            "lfr_raw": lfr_raw,
            "lfr_residual": lfr_res,
            "lfr_reduction": float(1.0 - lfr_res / (lfr_raw + epsilon)) if np.isfinite(lfr_raw) else np.nan,
            "variance_ratio": vr,
            "mean_residual": float(np.nanmean(residual)),
            "std_residual": float(np.nanstd(residual, ddof=1)) if len(part) > 1 else np.nan,
            "skewness": float(stats.skew(residual, nan_policy="omit")) if len(part) > 2 else np.nan,
            "kurtosis": float(stats.kurtosis(residual, nan_policy="omit")) if len(part) > 3 else np.nan,
            "acf_abs_lag66_sum": float(np.nansum(np.abs(acf66))),
            **lb,
        })
    return pd.DataFrame(rows)


def aggregate_diagnostics(by_ticker_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_split = by_ticker_split.groupby("base_split", dropna=False).agg(
        n=("n", "sum"),
        median_acf_energy_reduction=("acf_energy_reduction", "median"),
        median_lfr_reduction=("lfr_reduction", "median"),
        median_variance_ratio=("variance_ratio", "median"),
        max_abs_mean_residual=("mean_residual", lambda s: float(np.nanmax(np.abs(s)))),
    ).reset_index()
    overall = pd.DataFrame([{
        "n": int(by_ticker_split["n"].sum()),
        "median_acf_energy_reduction": float(by_ticker_split["acf_energy_reduction"].median()),
        "median_lfr_reduction": float(by_ticker_split["lfr_reduction"].median()),
        "median_variance_ratio": float(by_ticker_split["variance_ratio"].median()),
        "max_abs_mean_residual": float(by_ticker_split["mean_residual"].abs().max()),
    }])
    return overall, by_split
