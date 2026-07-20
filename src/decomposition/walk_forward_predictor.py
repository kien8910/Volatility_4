from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .p_model_loader import alpha_for


FEATURES = ["har_d", "har_w", "har_m"]


@dataclass(frozen=True)
class Segment:
    name: str
    fold_id: int
    origin_dates: list[pd.Timestamp]


def build_segments(panel: pd.DataFrame, split_manifest: pd.DataFrame, folds: pd.DataFrame, initial_training_days: int) -> list[Segment]:
    all_dates = sorted(split_manifest["date"].tolist())
    validation_dates = set(folds.loc[folds["role"].eq("validation"), "date"])
    test_dates = set(split_manifest.loc[split_manifest["is_locked_test"].eq(1), "date"])
    train_candidates = [d for d in all_dates if d not in validation_dates and d not in test_dates]
    train_origins = train_candidates[initial_training_days:]
    segments = [Segment("train", -1, train_origins)]
    for fold_id in sorted(folds["fold_id"].unique()):
        dates = sorted(folds.loc[folds["fold_id"].eq(fold_id) & folds["role"].eq("validation"), "date"].tolist())
        segments.append(Segment("validation", int(fold_id), dates))
    segments.append(Segment("test", 0, sorted(test_dates)))
    return segments


def _fit_predict(train: pd.DataFrame, row: pd.Series, target_col: str, alpha: float) -> tuple[float, int]:
    train = train.dropna(subset=FEATURES + [target_col])
    if len(train) < 30:
        return np.nan, 0
    model = Ridge(alpha=alpha)
    model.fit(train[FEATURES].to_numpy(dtype=float), train[target_col].to_numpy(dtype=float))
    pred = float(model.predict(row[FEATURES].to_numpy(dtype=float).reshape(1, -1))[0])
    return pred, len(train)


def walk_forward_predictions(panel: pd.DataFrame, split_manifest: pd.DataFrame, folds: pd.DataFrame, p_config: dict, horizons: list[int], initial_training_days: int, refit_frequency: int = 1) -> pd.DataFrame:
    rows = []
    segments = build_segments(panel, split_manifest, folds, initial_training_days)
    by_ticker = {ticker: part.sort_values("date").reset_index(drop=True) for ticker, part in panel.groupby("ticker", sort=False)}
    for ticker, part in by_ticker.items():
        for horizon in horizons:
            target_col = f"target_logvol_gk_h{horizon}"
            target_date_col = f"target_date_h{horizon}"
            valid_col = f"valid_origin_h{horizon}"
            alpha = alpha_for(p_config, ticker, horizon)
            origin_lookup = part.loc[part[valid_col]].set_index("date", drop=False)
            for seg in segments:
                for origin in seg.origin_dates:
                    if origin not in origin_lookup.index:
                        continue
                    row = origin_lookup.loc[origin]
                    train = part.loc[part[target_date_col].le(origin) & part[valid_col]]
                    pred, train_n = _fit_predict(train, row, target_col, alpha)
                    if not np.isfinite(pred):
                        continue
                    rows.append({
                        "date": row["date"],
                        "target_date": row[target_date_col],
                        "ticker": ticker,
                        "horizon": int(horizon),
                        "fold_id": seg.fold_id,
                        "base_split": seg.name,
                        "actual_target": float(row[target_col]),
                        "p_prediction": pred,
                        "residual_target": float(row[target_col] - pred),
                        "model_name": "HAR-Ridge",
                        "model_alpha": alpha,
                        "training_observations": int(train_n),
                        "is_oos": 1,
                        "max_training_target_date": train[target_date_col].max() if len(train) else pd.NaT,
                    })
    pred = pd.DataFrame(rows)
    if pred.empty:
        raise ValueError("No Step-3 out-of-sample predictions were generated.")
    pred["date"] = pd.to_datetime(pred["date"])
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred["max_training_target_date"] = pd.to_datetime(pred["max_training_target_date"])
    leak = pred.loc[pred["max_training_target_date"] > pred["date"]]
    if len(leak):
        raise ValueError("Leakage detected: training target date after forecast origin.")
    if not pred["target_date"].gt(pred["date"]).all():
        raise ValueError("Invalid target alignment: target_date must be greater than forecast origin date.")
    return pred.sort_values(["ticker", "horizon", "date"]).reset_index(drop=True)


def build_state_residuals(predictions: pd.DataFrame, panel: pd.DataFrame, split_labels: pd.DataFrame) -> pd.DataFrame:
    h1 = predictions.loc[predictions["horizon"].eq(1)].copy()
    state = h1.rename(columns={
        "date": "forecast_origin",
        "target_date": "date",
        "actual_target": "actual_logvol_gk",
        "p_prediction": "p_prediction_h1",
        "residual_target": "residual_state_h1",
    })
    state = state[["date", "ticker", "fold_id", "actual_logvol_gk", "p_prediction_h1", "residual_state_h1", "forecast_origin", "model_name", "is_oos", "max_training_target_date"]]
    state = state.merge(split_labels[["date", "base_split", "analysis_split"]], on="date", how="left", validate="many_to_one")
    state["base_split"] = state["analysis_split"].fillna(state["base_split"])
    state = state.drop(columns=["analysis_split"])
    return state.sort_values(["ticker", "date"]).reset_index(drop=True)
