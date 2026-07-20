from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from src.graph import SEMICONDUCTOR_TICKERS


class Step4DataError(ValueError):
    """Raised when Step 4 input data fail an explicit audit check."""


@dataclass(frozen=True)
class Step4Inputs:
    residual_state: pd.DataFrame
    residual_targets: pd.DataFrame
    p_predictions: pd.DataFrame
    split_manifest: pd.DataFrame | None
    fold_manifest: pd.DataFrame | None


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise Step4DataError(f"Missing required input file: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Step4DataError(f"Unsupported input format for {path}; expected .parquet or .csv")


def load_inputs(config: dict) -> Step4Inputs:
    data_cfg = config["data"]
    split_path = Path(data_cfg["split_manifest_path"])
    fold_path = Path(data_cfg["fold_manifest_path"])
    return Step4Inputs(
        residual_state=_read_table(data_cfg["residual_state_path"]),
        residual_targets=_read_table(data_cfg["residual_target_path"]),
        p_predictions=_read_table(data_cfg["p_prediction_path"]),
        split_manifest=_read_table(split_path) if split_path.exists() else None,
        fold_manifest=_read_table(fold_path) if fold_path.exists() else None,
    )


def _as_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _require_columns(df: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise Step4DataError(f"{name} is missing required columns: {missing}")


def _assert_finite(df: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    cols = list(columns)
    values = df[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = df.loc[~np.isfinite(values).all(axis=1), cols].head()
        raise Step4DataError(f"{name} contains NaN or infinite values in {cols}. Examples:\n{bad}")


def validate_inputs(inputs: Step4Inputs, tickers: list[str], horizons: list[int]) -> Step4Inputs:
    """Validate Step 3 artifacts before any model training.

    The checks here are intentionally strict. They stop execution before training if the
    residual branch and fixed P branch cannot be joined without ambiguity.
    """
    if tickers != SEMICONDUCTOR_TICKERS:
        raise Step4DataError(
            "Ticker order must exactly match SEMICONDUCTOR_TICKERS: "
            f"{SEMICONDUCTOR_TICKERS}. Received: {tickers}"
        )

    state = _as_datetime(inputs.residual_state, ["date", "forecast_origin"])
    targets = _as_datetime(inputs.residual_targets, ["date", "target_date"])
    preds = _as_datetime(inputs.p_predictions, ["date", "target_date"])

    _require_columns(
        state,
        "step3_residual_state",
        ["date", "ticker", "actual_logvol_gk", "residual_state_h1", "is_oos", "base_split"],
    )
    _require_columns(
        targets,
        "step3_residual_targets",
        [
            "date",
            "target_date",
            "ticker",
            "horizon",
            "base_split",
            "actual_target",
            "p_prediction",
            "residual_target",
            "is_oos",
        ],
    )
    _require_columns(
        preds,
        "oos_p_predictions",
        ["date", "target_date", "ticker", "horizon", "actual_target", "p_prediction", "residual_target", "is_oos"],
    )

    state_tickers = list(dict.fromkeys(state.sort_values(["date", "ticker"])["ticker"].tolist()))
    used_state_tickers = sorted(state["ticker"].unique().tolist())
    if used_state_tickers != sorted(tickers):
        raise Step4DataError(f"State data must contain exactly the 11 semiconductor tickers. Found: {used_state_tickers}")
    if sorted(targets["ticker"].unique().tolist()) != sorted(tickers):
        raise Step4DataError("Target data must contain exactly the 11 semiconductor tickers.")
    if sorted(preds["ticker"].unique().tolist()) != sorted(tickers):
        raise Step4DataError("P-prediction data must contain exactly the 11 semiconductor tickers.")

    if not set(targets["horizon"].astype(int).unique()).issubset(set(horizons)):
        raise Step4DataError("Target data contain horizons outside the configured horizon list.")
    if set(horizons) - set(targets["horizon"].astype(int).unique()):
        raise Step4DataError("Target data are missing one or more configured horizons.")

    for name, df in [("state", state), ("targets", targets), ("p_predictions", preds)]:
        if "is_oos" in df.columns and not (df["is_oos"].astype(int) == 1).all():
            raise Step4DataError(f"{name} contains rows where is_oos != 1.")

    dup_state = state.duplicated(["date", "ticker"])
    if dup_state.any():
        raise Step4DataError(f"Duplicate date x ticker rows in residual_state: {state.loc[dup_state].head()}")
    dup_targets = targets.duplicated(["date", "ticker", "horizon"])
    if dup_targets.any():
        raise Step4DataError(f"Duplicate date x ticker x horizon rows in residual_targets: {targets.loc[dup_targets].head()}")

    if not (targets["target_date"] > targets["date"]).all():
        bad = targets.loc[~(targets["target_date"] > targets["date"])].head()
        raise Step4DataError(f"Found target_date <= forecast_origin/date in residual_targets:\n{bad}")

    _assert_finite(state, "residual_state", ["actual_logvol_gk", "residual_state_h1"])
    _assert_finite(targets, "residual_targets", ["actual_target", "p_prediction", "residual_target"])
    _assert_finite(preds, "p_predictions", ["actual_target", "p_prediction", "residual_target"])

    join_cols = ["date", "target_date", "ticker", "horizon"]
    merged = targets.merge(
        preds[join_cols + ["actual_target", "p_prediction", "residual_target"]],
        on=join_cols,
        how="left",
        suffixes=("_target", "_p"),
    )
    if merged[["actual_target_p", "p_prediction_p", "residual_target_p"]].isna().any().any():
        raise Step4DataError("Residual target rows do not all have matching P-prediction rows.")
    for col in ["actual_target", "p_prediction", "residual_target"]:
        if not np.allclose(merged[f"{col}_target"], merged[f"{col}_p"], rtol=1e-9, atol=1e-9):
            raise Step4DataError(f"P branch mismatch for column {col}; residual target and P prediction artifacts disagree.")

    return Step4Inputs(
        residual_state=state.sort_values(["date", "ticker"]).reset_index(drop=True),
        residual_targets=targets.sort_values(["date", "horizon", "ticker"]).reset_index(drop=True),
        p_predictions=preds.sort_values(["date", "horizon", "ticker"]).reset_index(drop=True),
        split_manifest=inputs.split_manifest,
        fold_manifest=inputs.fold_manifest,
    )

