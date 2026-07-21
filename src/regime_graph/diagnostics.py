from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def regime_usage_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [col for col in predictions.columns if col.startswith("regime_weight_")]
    sample = predictions.drop_duplicates(["config_id", "split", "fold_id", "seed", "date"])[["config_id", "model", "split", "fold_id", "seed", "date", "regime_argmax"] + weight_cols]
    rows = []
    for keys, grp in sample.groupby(["config_id", "model", "split"], dropna=False):
        row = dict(zip(["config_id", "model", "split"], keys))
        weights = grp[weight_cols].to_numpy(dtype=float)
        valid_cols = ~np.isnan(weights).all(axis=0)
        weights = weights[:, valid_cols]
        if weights.size == 0:
            row.update({"gate_entropy": np.nan, "effective_regimes": 1.0, "collapse_90": False, "collapse_95": False})
        else:
            mean_w = np.nanmean(weights, axis=0)
            entropy = float(-(mean_w * np.log(mean_w + 1e-12)).sum())
            row.update(
                {
                    "gate_entropy": entropy,
                    "effective_regimes": float(np.exp(entropy)),
                    "collapse_90": bool(np.nanmax(mean_w) > 0.90),
                    "collapse_95": bool(np.nanmax(mean_w) > 0.95),
                }
            )
            for i, value in enumerate(mean_w, start=1):
                row[f"mean_regime_weight_{i}"] = float(value)
                row[f"p_regime_weight_{i}_gt_0_5"] = float(np.nanmean(weights[:, i - 1] > 0.5))
        rows.append(row)
    return pd.DataFrame(rows)


def graph_edges_from_bank(adj_bank: torch.Tensor, tickers: list[str], run_config: dict, fold_id: int, seed: int) -> pd.DataFrame:
    rows = []
    arr = adj_bank.detach().cpu().numpy()
    for k in range(arr.shape[0]):
        for i, src in enumerate(tickers):
            for j, dst in enumerate(tickers):
                if i != j and arr[k, i, j] > 0:
                    rows.append(
                        {
                            "config_id": run_config["config_id"],
                            "model": run_config["model"],
                            "K": run_config["K"],
                            "ema_beta": run_config["ema_beta"],
                            "fold_id": int(fold_id),
                            "seed": int(seed),
                            "regime": int(k + 1),
                            "source": src,
                            "target": dst,
                            "weight": float(arr[k, i, j]),
                        }
                    )
    return pd.DataFrame(rows)


def graph_diversity_from_bank(adj_bank: torch.Tensor, run_config: dict, fold_id: int, seed: int) -> pd.DataFrame:
    arr = adj_bank.detach().cpu().numpy()
    rows = []
    for a, b in combinations(range(arr.shape[0]), 2):
        va = arr[a].reshape(-1)
        vb = arr[b].reshape(-1)
        top_a = set(np.flatnonzero(va > 0).tolist())
        top_b = set(np.flatnonzero(vb > 0).tolist())
        denom = len(top_a | top_b) or 1
        rows.append(
            {
                "config_id": run_config["config_id"],
                "model": run_config["model"],
                "fold_id": int(fold_id),
                "seed": int(seed),
                "regime_a": int(a + 1),
                "regime_b": int(b + 1),
                "frobenius_distance": float(np.linalg.norm(arr[a] - arr[b])),
                "cosine_similarity": float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12)),
                "spearman_correlation": float(spearmanr(va, vb).correlation) if np.std(va) > 0 and np.std(vb) > 0 else np.nan,
                "topk_jaccard": float(len(top_a & top_b) / denom),
            }
        )
    return pd.DataFrame(rows)

