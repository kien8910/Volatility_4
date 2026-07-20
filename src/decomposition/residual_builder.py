from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEMICONDUCTOR_TICKERS = [
    "ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC",
    "LRCX", "MU", "NVDA", "QCOM", "TXN",
]


def load_panel(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = [str(t).upper() for t in config["data"]["tickers"]]
    if tickers != SEMICONDUCTOR_TICKERS:
        raise ValueError("Step-3 must use exactly the fixed 11 semiconductor tickers in order.")
    horizons = [int(h) for h in config["target"]["horizons"]]
    cols = ["date", "ticker", "open", "high", "low", "close", "ohlc_valid", "log_return", "logvol_gk", "base_split", "is_locked_test"]
    for h in horizons:
        cols.extend([f"target_date_h{h}", f"target_logvol_gk_h{h}"])
    panel = pd.read_parquet(root / config["data"]["panel_path"], columns=cols)
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    missing = sorted(set(tickers) - set(panel["ticker"].unique()))
    if missing:
        raise ValueError(f"Missing required tickers: {missing}")
    panel = panel.loc[panel["ticker"].isin(tickers)].sort_values(["ticker", "date"]).reset_index(drop=True)
    if panel["ticker"].drop_duplicates().tolist() != tickers:
        present = panel["ticker"].drop_duplicates().tolist()
        raise ValueError(f"Unexpected ticker order/content: {present}")
    if not panel[["ticker", "date"]].equals(panel.sort_values(["ticker", "date"])[["ticker", "date"]].reset_index(drop=True)):
        raise ValueError("Panel must be sorted by ticker/date.")
    split_manifest = pd.read_csv(root / config["data"]["split_manifest_path"], parse_dates=["date"])
    folds = pd.read_csv(root / config["data"]["expanding_folds_path"], parse_dates=["date"])
    return panel, split_manifest, folds


def add_har_features(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = panel.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", group_keys=False)
    out["har_d"] = out["logvol_gk"]
    out["har_w"] = g["logvol_gk"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    out["har_m"] = g["logvol_gk"].rolling(22, min_periods=22).mean().reset_index(level=0, drop=True)
    for h in horizons:
        out[f"valid_origin_h{h}"] = (
            out["ohlc_valid"].astype(bool)
            & out["logvol_gk"].notna()
            & out["har_m"].notna()
            & out[f"target_logvol_gk_h{h}"].notna()
            & out[f"target_date_h{h}"].notna()
            & (out[f"target_date_h{h}"] > out["date"])
        )
    return out


def date_split_labels(split_manifest: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    labels = split_manifest.copy()
    validation_dates = set(folds.loc[folds["role"].eq("validation"), "date"])
    labels["analysis_split"] = np.where(labels["is_locked_test"].eq(1), "test", np.where(labels["date"].isin(validation_dates), "validation", "train"))
    return labels[["date", "base_split", "is_locked_test", "analysis_split"]]


def qlike_from_logvol(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, epsilon: float) -> np.ndarray:
    true_var = np.maximum(np.exp(np.clip(2.0 * np.asarray(y_true, dtype=float), -60, 60)), epsilon)
    pred_var = np.maximum(np.exp(np.clip(2.0 * np.asarray(y_pred, dtype=float), -60, 60)), epsilon)
    ratio = true_var / pred_var
    return ratio - np.log(ratio) - 1.0
