from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.graph.panel_builder import GraphSampleTable


@dataclass(frozen=True)
class DatasetMeta:
    indices: np.ndarray
    dates: np.ndarray
    target_dates: np.ndarray
    split: np.ndarray
    fold_id: np.ndarray


class GraphWindowDataset(Dataset):
    def __init__(self, samples: GraphSampleTable, indices: np.ndarray, input_kind: str, x_scaled: np.ndarray | None = None):
        if input_kind not in {"residual", "raw"}:
            raise ValueError("input_kind must be 'residual' or 'raw'")
        self.samples = samples
        self.indices = np.asarray(indices, dtype=np.int64)
        source = samples.residual_windows if input_kind == "residual" else samples.raw_windows
        self.x = source if x_scaled is None else x_scaled
        self.y_residual = samples.target_residual
        self.y_actual = samples.target_actual
        self.p_prediction = samples.p_prediction
        self.meta = DatasetMeta(
            indices=self.indices,
            dates=samples.sample_dates.to_numpy()[self.indices],
            target_dates=samples.target_dates[self.indices],
            split=samples.split[self.indices],
            fold_id=samples.fold_id[self.indices],
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        source_idx = self.indices[idx]
        return {
            "x": torch.from_numpy(self.x[source_idx]).float(),
            "y_residual": torch.from_numpy(self.y_residual[source_idx]).float(),
            "y_actual": torch.from_numpy(self.y_actual[source_idx]).float(),
            "p_prediction": torch.from_numpy(self.p_prediction[source_idx]).float(),
            "sample_index": torch.tensor(source_idx, dtype=torch.long),
        }


def make_scaled_datasets(samples: GraphSampleTable, train_idx: np.ndarray, eval_idx: np.ndarray, input_kind: str, scaler):
    source = samples.residual_windows if input_kind == "residual" else samples.raw_windows
    scaler.fit(source[train_idx])
    scaled = scaler.transform(source)
    return (
        GraphWindowDataset(samples, train_idx, input_kind=input_kind, x_scaled=scaled),
        GraphWindowDataset(samples, eval_idx, input_kind=input_kind, x_scaled=scaled),
        scaler,
    )

