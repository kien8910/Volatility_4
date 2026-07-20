from __future__ import annotations

import numpy as np
import pandas as pd


def qlike_from_logvol(
    actual_logvol: np.ndarray,
    predicted_logvol: np.ndarray,
    epsilon: float = 1e-12,
    clip_min: float = -20.0,
    clip_max: float = 20.0,
) -> tuple[np.ndarray, int]:
    actual = np.asarray(actual_logvol, dtype=float)
    pred = np.asarray(predicted_logvol, dtype=float)
    clipped_actual = np.clip(actual, clip_min, clip_max)
    clipped_pred = np.clip(pred, clip_min, clip_max)
    clipped_count = int(np.sum(actual != clipped_actual) + np.sum(pred != clipped_pred))
    sigma2 = np.maximum(np.exp(2.0 * clipped_actual), epsilon)
    sigma2_hat = np.maximum(np.exp(2.0 * clipped_pred), epsilon)
    ratio = sigma2 / sigma2_hat
    return ratio - np.log(ratio) - 1.0, clipped_count


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def summarize_predictions(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": int(len(grp)),
                "qlike": float(grp["qlike_loss"].mean()),
                "residual_mse": mse(grp["residual_actual"], grp["residual_prediction"]),
                "residual_mae": mae(grp["residual_actual"], grp["residual_prediction"]),
                "final_mae": mae(grp["actual_logvol"], grp["final_prediction"]),
                "spike_qlike": float(grp.loc[grp["spike_flag"].astype(bool), "qlike_loss"].mean())
                if grp["spike_flag"].astype(bool).any()
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

