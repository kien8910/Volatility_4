from __future__ import annotations

import pandas as pd


def simple_treatment_control_did(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.loc[frame["is_direct_target"].astype(int).eq(1)].copy()
    if target.empty:
        return pd.DataFrame(columns=["horizon", "hierarchy", "mean_did", "n"])
    control = frame.loc[frame["is_direct_target"].astype(int).eq(0)].copy()
    if control.empty:
        return pd.DataFrame(columns=["horizon", "hierarchy", "mean_did", "n"])
    cmean = control.groupby(["date", "horizon"], as_index=False)["abnormal_volatility_response"].mean().rename(
        columns={"abnormal_volatility_response": "control_av"}
    )
    merged = target.merge(cmean, on=["date", "horizon"], how="left")
    merged["did"] = merged["abnormal_volatility_response"] - merged["control_av"]
    return merged.groupby(["horizon", "hierarchy"], as_index=False).agg(mean_did=("did", "mean"), n=("did", "size"))
