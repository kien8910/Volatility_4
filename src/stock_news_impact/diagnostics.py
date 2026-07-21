from __future__ import annotations

import numpy as np
import pandas as pd


def gate_diagnostics(gates: pd.DataFrame) -> pd.DataFrame:
    if gates.empty:
        return pd.DataFrame(columns=["model", "hierarchy", "horizon", "mean_gate", "median_gate", "n"])
    return (
        gates.groupby(["model", "hierarchy", "horizon"], as_index=False)
        .agg(mean_gate=("final_gate", "mean"), median_gate=("final_gate", "median"), n=("final_gate", "size"))
        .sort_values(["model", "hierarchy", "horizon"])
    )


def common_news_impact_diagnostics(gates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether macro/sector events move semiconductor stocks coherently.

    For common-scope news, impact evidence should not be judged stock-by-stock only.
    This diagnostic summarizes whether the gated correction has a shared sign and
    meaningful cross-sectional average across the 11-stock semiconductor universe.
    It is descriptive, not causal.
    """
    columns = [
        "model",
        "config_id",
        "event_id",
        "date",
        "hierarchy",
        "horizon",
        "fold_id",
        "seed",
        "n_stocks",
        "mean_gated_correction",
        "mean_abs_gated_correction",
        "std_gated_correction",
        "commonality_ratio",
        "same_sign_rate",
        "mean_utility",
        "mean_abnormal_response",
    ]
    if gates.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, pd.DataFrame()
    common = gates.loc[gates["hierarchy"].astype(str).isin(["macro", "sector"])].copy()
    if common.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, pd.DataFrame()
    rows = []
    group_cols = ["model", "config_id", "event_id", "date", "hierarchy", "horizon", "fold_id", "seed"]
    for keys, grp in common.groupby(group_cols, dropna=False):
        corr = grp["gated_correction"].astype(float).to_numpy()
        mean_abs = float(np.mean(np.abs(corr))) if len(corr) else 0.0
        pos_rate = float(np.mean(corr > 0)) if len(corr) else 0.0
        neg_rate = float(np.mean(corr < 0)) if len(corr) else 0.0
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_stocks": int(grp["target_ticker"].nunique()),
                "mean_gated_correction": float(np.mean(corr)) if len(corr) else 0.0,
                "mean_abs_gated_correction": mean_abs,
                "std_gated_correction": float(np.std(corr)) if len(corr) else 0.0,
                "commonality_ratio": float(abs(np.mean(corr)) / mean_abs) if mean_abs > 0 else 0.0,
                "same_sign_rate": max(pos_rate, neg_rate),
                "mean_utility": float(grp["utility"].astype(float).mean()) if "utility" in grp else np.nan,
                "mean_abnormal_response": float(grp["abnormal_volatility_response"].astype(float).mean())
                if "abnormal_volatility_response" in grp
                else np.nan,
            }
        )
    detail = pd.DataFrame(rows, columns=columns)
    summary = (
        detail.groupby(["model", "config_id", "hierarchy", "horizon"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_common_impact=("mean_gated_correction", "mean"),
            mean_abs_common_impact=("mean_gated_correction", lambda x: float(np.mean(np.abs(x)))),
            mean_cross_stock_abs_impact=("mean_abs_gated_correction", "mean"),
            mean_commonality_ratio=("commonality_ratio", "mean"),
            mean_same_sign_rate=("same_sign_rate", "mean"),
            mean_utility=("mean_utility", "mean"),
            mean_abnormal_response=("mean_abnormal_response", "mean"),
        )
        .sort_values(["model", "hierarchy", "horizon"])
    )
    return detail, summary
