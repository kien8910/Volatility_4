from __future__ import annotations

import numpy as np


def acf_values(x, max_lag: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= max_lag + 1:
        return np.full(max_lag, np.nan)
    arr = arr - arr.mean()
    denom = float(np.dot(arr, arr))
    if denom <= 0:
        return np.zeros(max_lag)
    return np.array([float(np.dot(arr[lag:], arr[:-lag]) / denom) for lag in range(1, max_lag + 1)])


def acf_energy(x, lag: int) -> float:
    vals = acf_values(x, lag)
    return float(np.nansum(np.abs(vals)))


def low_frequency_ratio(x, fraction: float = 0.10, epsilon: float = 1.0e-12) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 8:
        return np.nan
    arr = arr - arr.mean()
    power = np.abs(np.fft.rfft(arr)) ** 2
    positive = power[1:]
    if len(positive) == 0:
        return np.nan
    low_n = max(1, int(np.ceil(len(positive) * fraction)))
    return float(positive[:low_n].sum() / (positive.sum() + epsilon))
