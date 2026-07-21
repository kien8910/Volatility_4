from __future__ import annotations

import numpy as np
import pandas as pd


def news_correction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cols in [["split", "model"], ["split", "model", "ticker"], ["split", "model", "horizon"]]:
        for keys, grp in predictions.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(cols, keys))
            corr = grp["news_residual_correction"].to_numpy(dtype=float)
            row.update(
                {
                    "grouping": "+".join(cols),
                    "n": int(len(grp)),
                    "mean_abs_correction": float(np.mean(np.abs(corr))),
                    "near_zero_rate": float(np.mean(np.abs(corr) < 1e-6)),
                    "extreme_rate": float(np.mean(np.abs(corr) > np.quantile(np.abs(corr), 0.99))) if len(corr) > 1 else 0.0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def embedding_statistics(embedding_frame: pd.DataFrame) -> pd.DataFrame:
    if embedding_frame.empty:
        return pd.DataFrame(columns=["encoder_name", "pooling_method", "hierarchy", "rows", "unique_text_hashes", "embedding_dim", "nonfinite_rows"])
    rows = []
    for keys, grp in embedding_frame.groupby(["encoder_name", "pooling_method", "hierarchy"], dropna=False):
        bad = 0
        for emb in grp["embedding"]:
            arr = np.asarray(emb, dtype=np.float32)
            bad += int(not np.isfinite(arr).all())
        rows.append(
            {
                "encoder_name": keys[0],
                "pooling_method": keys[1],
                "hierarchy": keys[2],
                "rows": int(len(grp)),
                "unique_text_hashes": int(grp["text_hash"].nunique()),
                "embedding_dim": int(grp["embedding_dim"].iloc[0]),
                "nonfinite_rows": int(bad),
            }
        )
    return pd.DataFrame(rows)

