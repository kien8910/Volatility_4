from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class EvalConfig:
    epsilon: float = 1.0e-12
    exp_clip: tuple[float, float] = (-60.0, 60.0)


def qlike(y_true_logvol, y_pred_logvol, epsilon: float = 1.0e-12, exp_clip=(-60.0, 60.0)):
    y_true = np.asarray(y_true_logvol, dtype=float)
    y_pred = np.asarray(y_pred_logvol, dtype=float)
    pred_exp = np.clip(2.0 * y_pred, exp_clip[0], exp_clip[1])
    true_exp = np.clip(2.0 * y_true, exp_clip[0], exp_clip[1])
    sigma2 = np.maximum(np.exp(true_exp), epsilon)
    sigma2_hat = np.maximum(np.exp(pred_exp), epsilon)
    ratio = sigma2 / sigma2_hat
    loss = ratio - np.log(ratio) - 1.0
    clipped = (2.0 * y_pred < exp_clip[0]) | (2.0 * y_pred > exp_clip[1])
    return loss, clipped


def prediction_metrics(df: pd.DataFrame, spike_quantile: float, epsilon: float, exp_clip=(-60.0, 60.0)) -> dict[str, float]:
    if len(df) == 0:
        return {}
    y = df["y_true"].to_numpy(dtype=float)
    pred = df["y_pred"].to_numpy(dtype=float)
    err = pred - y
    ql, clipped = qlike(y, pred, epsilon, exp_clip)
    spike = df["is_spike"].to_numpy(dtype=bool) if "is_spike" in df else np.zeros(len(df), dtype=bool)
    pearson = np.nan
    spearman = np.nan
    if len(df) > 1 and np.std(y) > 0 and np.std(pred) > 0:
        pearson = float(stats.pearsonr(y, pred).statistic)
        spearman = float(stats.spearmanr(y, pred).statistic)
    return {
        "n": int(len(df)),
        "qlike": float(np.mean(ql)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": pearson,
        "spearman": spearman,
        "qlike_spike_days": float(np.mean(ql[spike])) if spike.any() else np.nan,
        "qlike_normal_days": float(np.mean(ql[~spike])) if (~spike).any() else np.nan,
        "pred_exp_clip_count": int(clipped.sum()),
    }


def summarize_predictions(pred: pd.DataFrame, group_cols: list[str], spike_quantile: float, epsilon: float, exp_clip=(-60.0, 60.0)) -> pd.DataFrame:
    rows = []
    for keys, part in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(prediction_metrics(part, spike_quantile, epsilon, exp_clip))
        row["training_time_sec"] = float(part["training_time_sec"].mean()) if "training_time_sec" in part else np.nan
        row["inference_time_sec"] = float(part["inference_time_sec"].sum()) if "inference_time_sec" in part else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, block_lag: int = 5) -> tuple[float, float]:
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    mean_d = d.mean()
    centered = d - mean_d
    gamma0 = np.mean(centered * centered)
    var = gamma0
    lag = min(block_lag, n - 1)
    for k in range(1, lag + 1):
        gamma = np.mean(centered[k:] * centered[:-k])
        var += 2.0 * (1.0 - k / (lag + 1)) * gamma
    var = max(var / n, 1.0e-18)
    stat = mean_d / np.sqrt(var)
    pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pvalue)


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.dropna().sort_values()
    m = len(valid)
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    running = 0.0
    for rank, (idx, pvalue) in enumerate(valid.items(), start=1):
        running = max(running, min((m - rank + 1) * pvalue, 1.0))
        adjusted.loc[idx] = running
    return adjusted
