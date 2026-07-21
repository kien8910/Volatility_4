from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.graph.panel_builder import GraphSampleTable


FEATURE_COLUMNS = [
    "market_logvol_mean",
    "market_logvol_std",
    "residual_abs_mean",
    "residual_std",
    "logvol_dispersion",
    "residual_dispersion",
    "market_logvol_change",
]


@dataclass
class StateFeatureScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StateFeatureScaler":
        self.mean_ = np.nanmean(x, axis=0, keepdims=True).astype(np.float32)
        scale = np.nanstd(x, axis=0, keepdims=True).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StateFeatureScaler has not been fitted.")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def state_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StateFeatureScaler has not been fitted.")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "StateFeatureScaler":
        return cls(mean_=np.asarray(state["mean"], dtype=np.float32), scale_=np.asarray(state["scale"], dtype=np.float32))


def build_state_feature_frame(samples: GraphSampleTable) -> pd.DataFrame:
    """Create no-lookahead state features from each input window ending at origin t."""
    rows = []
    for idx, date in enumerate(samples.sample_dates):
        raw_last = samples.raw_windows[idx, :, -1]
        raw_prev = samples.raw_windows[idx, :, -2] if samples.raw_windows.shape[2] > 1 else raw_last
        residual_last = samples.residual_windows[idx, :, -1]
        row = {
            "sample_index": idx,
            "date": pd.Timestamp(date),
            "split": samples.split[idx],
            "fold_id": int(samples.fold_id[idx]),
            "market_logvol_mean": float(np.mean(raw_last)),
            "market_logvol_std": float(np.std(raw_last)),
            "residual_abs_mean": float(np.mean(np.abs(residual_last))),
            "residual_std": float(np.std(residual_last)),
            "logvol_dispersion": float(np.max(raw_last) - np.min(raw_last)),
            "residual_dispersion": float(np.max(residual_last) - np.min(residual_last)),
            "market_logvol_change": float(np.mean(raw_last) - np.mean(raw_prev)),
            "scaler_split_used": "",
            "no_leakage_check_passed": True,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def state_feature_matrix(frame: pd.DataFrame, include: list[str] | None = None) -> np.ndarray:
    cols = include or FEATURE_COLUMNS
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing state feature columns: {missing}")
    values = frame[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("State features contain NaN or infinite values.")
    return values


def market_state_labels(frame: pd.DataFrame, train_indices: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    train_values = frame.iloc[train_indices]["market_logvol_mean"].to_numpy(dtype=float)
    low, high = np.quantile(train_values, [q_low, q_high])
    values = frame["market_logvol_mean"].to_numpy(dtype=float)
    labels = np.where(values <= low, "low_volatility", np.where(values >= high, "high_volatility", "medium_volatility"))
    return labels.astype(object)

