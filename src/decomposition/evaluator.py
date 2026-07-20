from __future__ import annotations

import pandas as pd

from .residual_builder import qlike_from_logvol


def p_model_performance(targets: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows = []
    for keys, part in targets.groupby(["base_split", "ticker", "horizon"], dropna=False):
        split, ticker, horizon = keys
        q = qlike_from_logvol(part["actual_target"], part["p_prediction"], epsilon)
        err = part["p_prediction"].to_numpy(dtype=float) - part["actual_target"].to_numpy(dtype=float)
        rows.append({
            "base_split": split,
            "ticker": ticker,
            "horizon": int(horizon),
            "n": int(len(part)),
            "qlike": float(q.mean()),
            "mse": float((err ** 2).mean()),
            "mae": float(abs(err).mean()),
            "mean_residual": float(part["residual_target"].mean()),
        })
    return pd.DataFrame(rows)
