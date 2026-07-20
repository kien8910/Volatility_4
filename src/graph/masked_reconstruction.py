from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.graph.losses import residual_loss
from src.graph.metrics import mae, mse
from src.graph.models import MaskedReconstructionModel


class ReconstructionDataset(Dataset):
    def __init__(self, x: np.ndarray, indices: np.ndarray, mask_ratio: float, seed: int):
        self.x = x
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mask_ratio = float(mask_ratio)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        source_idx = int(self.indices[idx])
        x = torch.from_numpy(self.x[source_idx]).float()
        target = x[:, -1].clone()
        rng = np.random.default_rng(self.seed + source_idx)
        n = x.shape[0]
        k = max(1, int(round(n * self.mask_ratio)))
        masked = rng.choice(n, size=k, replace=False)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[masked] = True
        return {"x": x, "target": target, "mask": mask}


@dataclass(frozen=True)
class ReconstructionConfig:
    model_id: str
    graph_type: str
    fixed_adjacency: torch.Tensor | None
    top_k: int | None = None
    graph_embedding_dim: int | None = None
    directed: bool = False


def run_reconstruction_diagnostic(
    x_scaled: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    tickers: list[str],
    temporal_kind: str,
    temporal_cfg: dict,
    configs: list[ReconstructionConfig],
    mask_ratios: list[float],
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int = 20,
) -> pd.DataFrame:
    rows = []
    for mask_ratio in mask_ratios:
        train_ds = ReconstructionDataset(x_scaled, train_idx, mask_ratio, seed)
        eval_ds = ReconstructionDataset(x_scaled, eval_idx, mask_ratio, seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        for rcfg in configs:
            model = MaskedReconstructionModel(
                num_nodes=len(tickers),
                lookback=x_scaled.shape[2],
                temporal_kind=temporal_kind,
                temporal_cfg=temporal_cfg,
                graph_type=rcfg.graph_type,
                fixed_adjacency=rcfg.fixed_adjacency,
                graph_embedding_dim=rcfg.graph_embedding_dim,
                top_k=rcfg.top_k,
                directed=rcfg.directed,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(max_epochs):
                model.train()
                for batch in train_loader:
                    opt.zero_grad(set_to_none=True)
                    x = batch["x"].to(device)
                    mask = batch["mask"].to(device)
                    target = batch["target"].to(device)
                    pred = model(x, mask)
                    loss = residual_loss(pred[mask], target[mask], "mse")
                    loss.backward()
                    opt.step()
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for batch in eval_loader:
                    x = batch["x"].to(device)
                    mask = batch["mask"].to(device)
                    target = batch["target"].to(device)
                    pred = model(x, mask)
                    preds.append(pred[mask].detach().cpu().numpy())
                    actuals.append(target[mask].detach().cpu().numpy())
            pred_arr = np.concatenate(preds)
            actual_arr = np.concatenate(actuals)
            rows.append(
                {
                    "model": rcfg.model_id,
                    "graph_type": rcfg.graph_type,
                    "mask_ratio": mask_ratio,
                    "seed": seed,
                    "mse": mse(actual_arr, pred_arr),
                    "mae": mae(actual_arr, pred_arr),
                }
            )
    return pd.DataFrame(rows)

