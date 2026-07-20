from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SEMICONDUCTOR_TICKERS = [
    "ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC",
    "LRCX", "MU", "NVDA", "QCOM", "TXN",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def recompute_gk(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    out = df.copy()
    log_hl = np.log(out["high"] / out["low"])
    log_co = np.log(out["close"] / out["open"])
    out["gk_variance_raw_recomputed"] = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
    out["gk_variance_recomputed"] = np.maximum(out["gk_variance_raw_recomputed"], epsilon)
    out["logvol_gk_recomputed"] = 0.5 * np.log(out["gk_variance_recomputed"])
    return out


def validate_target_consistency(df: pd.DataFrame, epsilon: float, tol: float = 1e-10) -> dict[str, Any]:
    checked = recompute_gk(df, epsilon)
    diff = (checked["logvol_gk"] - checked["logvol_gk_recomputed"]).abs()
    max_abs_error = float(diff.max())
    if not np.isfinite(max_abs_error) or max_abs_error > tol:
        raise ValueError(f"Step-0 logvol_gk is inconsistent with GK formula: max_abs_error={max_abs_error}")
    nonpositive = int((checked["gk_variance_raw_recomputed"] <= 0).sum())
    return {
        "target_check_rows": int(len(checked)),
        "max_abs_logvol_gk_error": max_abs_error,
        "gk_nonpositive_count": nonpositive,
    }


def load_step1_inputs(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = [str(t).upper() for t in config["data"]["tickers"]]
    if tickers != SEMICONDUCTOR_TICKERS:
        raise ValueError("Config tickers must exactly match the 11 fixed semiconductor tickers in order.")
    panel = pd.read_parquet(root / config["data"]["panel_path"])
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    missing = sorted(set(tickers) - set(panel["ticker"].unique()))
    if missing:
        raise ValueError(f"Missing required semiconductor tickers: {missing}")
    keep_cols = [
        "date", "ticker", "open", "high", "low", "close", "ohlc_valid",
        "log_return", "squared_return", "absolute_return",
        "gk_variance_raw", "gk_nonpositive_flag", "gk_variance", "logvol_gk",
        "rs_variance_raw", "rs_nonpositive_flag", "rs_variance", "logvol_rs",
        "base_split", "is_locked_test",
    ]
    for h in config["target"]["horizons"]:
        keep_cols += [f"target_date_h{h}", f"target_logvol_gk_h{h}"]
    df = panel.loc[panel["ticker"].isin(tickers), keep_cols].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    target_check = validate_target_consistency(df, float(config["target"]["epsilon"]))
    split_manifest = pd.read_csv(root / config["data"]["split_manifest_path"], parse_dates=["date"])
    folds = pd.read_csv(root / config["data"]["expanding_folds_path"], parse_dates=["date"])
    counts = df.groupby("ticker").agg(
        observations=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        missing_ohlc=("ohlc_valid", lambda s: int((~s.astype(bool)).sum())),
        missing_logvol=("logvol_gk", lambda s: int(s.isna().sum())),
    ).reset_index()
    return df, split_manifest, folds, {**target_check, "ticker_counts": counts}


def add_time_series_features(df: pd.DataFrame, lookbacks: list[int], horizons: list[int]) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", group_keys=False)
    out["har_d"] = out["logvol_gk"]
    out["har_w"] = g["logvol_gk"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    out["har_m"] = g["logvol_gk"].rolling(22, min_periods=22).mean().reset_index(level=0, drop=True)
    max_lb = max(lookbacks)
    for lag in range(max_lb):
        out[f"lag_{lag}"] = g["logvol_gk"].shift(lag)
    for h in horizons:
        out[f"valid_h{h}"] = (
            out["ohlc_valid"].astype(bool)
            & out["logvol_gk"].notna()
            & out[f"target_logvol_gk_h{h}"].notna()
            & out[f"target_date_h{h}"].notna()
            & (out[f"target_date_h{h}"] > out["date"])
            & out["har_m"].notna()
            & out[[f"lag_{i}" for i in range(max_lb)]].notna().all(axis=1)
        )
    return out


def split_dates_for_fold(folds: pd.DataFrame, fold_id: int) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    fold = folds.loc[folds["fold_id"].eq(fold_id)]
    train_dates = set(fold.loc[fold["role"].eq("train"), "date"])
    val_dates = set(fold.loc[fold["role"].eq("validation"), "date"])
    if train_dates & val_dates:
        raise ValueError(f"Train/validation overlap in fold {fold_id}")
    return train_dates, val_dates


def dev_test_dates(split_manifest: pd.DataFrame) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    dev = set(split_manifest.loc[split_manifest["is_locked_test"].eq(0), "date"])
    test = set(split_manifest.loc[split_manifest["is_locked_test"].eq(1), "date"])
    if dev & test:
        raise ValueError("Development and locked test dates overlap.")
    return dev, test
