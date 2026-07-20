from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler3D:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler3D":
        if x.ndim != 3:
            raise ValueError(f"Expected [samples, tickers, lookback], got shape {x.shape}")
        self.mean_ = np.nanmean(x, axis=(0, 2), keepdims=True).astype(np.float32)
        scale = np.nanstd(x, axis=(0, 2), keepdims=True).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "StandardScaler3D":
        return cls(mean_=np.asarray(state["mean"], dtype=np.float32), scale_=np.asarray(state["scale"], dtype=np.float32))

