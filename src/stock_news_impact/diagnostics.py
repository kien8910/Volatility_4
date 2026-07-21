from __future__ import annotations

import pandas as pd


def gate_diagnostics(gates: pd.DataFrame) -> pd.DataFrame:
    if gates.empty:
        return pd.DataFrame(columns=["model", "hierarchy", "horizon", "mean_gate", "median_gate", "n"])
    return (
        gates.groupby(["model", "hierarchy", "horizon"], as_index=False)
        .agg(mean_gate=("final_gate", "mean"), median_gate=("final_gate", "median"), n=("final_gate", "size"))
        .sort_values(["model", "hierarchy", "horizon"])
    )
