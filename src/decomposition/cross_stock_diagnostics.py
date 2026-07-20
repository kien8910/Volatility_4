from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def residual_correlation_matrix(state: pd.DataFrame) -> pd.DataFrame:
    train = state.loc[state["base_split"].eq("train")]
    pivot = train.pivot_table(index="date", columns="ticker", values="residual_state_h1")
    return pivot.corr()


def raw_correlation_matrix(state: pd.DataFrame) -> pd.DataFrame:
    train = state.loc[state["base_split"].eq("train")]
    pivot = train.pivot_table(index="date", columns="ticker", values="actual_logvol_gk")
    return pivot.corr()


def lagged_cross_correlations(state: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    train = state.loc[state["base_split"].eq("train")]
    pivot = train.pivot_table(index="date", columns="ticker", values="residual_state_h1").sort_index()
    rows = []
    tickers = list(pivot.columns)
    for lag in lags:
        shifted = pivot.shift(lag)
        for src in tickers:
            for dst in tickers:
                if src == dst:
                    continue
                pair = pd.concat([pivot[dst], shifted[src]], axis=1).dropna()
                if len(pair) < 10:
                    corr, pvalue = np.nan, np.nan
                else:
                    corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                    if np.isfinite(corr) and abs(corr) < 1:
                        t = corr * np.sqrt((len(pair) - 2) / max(1.0 - corr ** 2, 1.0e-12))
                        pvalue = float(2 * (1 - stats.t.cdf(abs(t), df=len(pair) - 2)))
                    else:
                        pvalue = np.nan
                rows.append({"source_ticker": src, "target_ticker": dst, "lag": int(lag), "correlation": corr, "pvalue": pvalue, "n": int(len(pair))})
    return pd.DataFrame(rows)


def dependency_summary(corr: pd.DataFrame, raw_corr: pd.DataFrame, lagged: pd.DataFrame) -> dict:
    off = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    raw_off = raw_corr.where(~np.eye(len(raw_corr), dtype=bool)).stack()
    return {
        "mean_abs_offdiag_residual_corr": float(off.abs().mean()) if len(off) else np.nan,
        "mean_abs_offdiag_raw_corr": float(raw_off.abs().mean()) if len(raw_off) else np.nan,
        "significant_lagged_pairs_p05": int((lagged["pvalue"] < 0.05).sum()) if len(lagged) else 0,
        "max_abs_lagged_corr": float(lagged["correlation"].abs().max()) if len(lagged) else np.nan,
    }
