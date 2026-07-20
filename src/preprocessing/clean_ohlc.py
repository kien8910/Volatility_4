"""Create auditable OHLC validity and corporate-action candidate flags."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .common import OHLC, setup_logging

def audit_ohlc(df: pd.DataFrame, extreme: float=.35) -> pd.DataFrame:
    x=df.copy(); vals=x[OHLC].to_numpy(float)
    x["invalid_nonpositive_price"]=(x[OHLC]<=0).any(axis=1)
    x["invalid_nonfinite_price"]=~np.isfinite(vals).all(axis=1)
    x["invalid_high_rule"]=x.high < x[["open","close","low"]].max(axis=1)
    x["invalid_low_rule"]=x.low > x[["open","close","high"]].min(axis=1)
    x["invalid_high_low"]=x.high < x.low
    prev=x.sort_values(["ticker","date"]).groupby("ticker").close.shift()
    x["close_log_return_raw"]=np.log(x.close/prev)
    x["extreme_close_return"]=x.close_log_return_raw.abs()>extreme
    ratios=x[OHLC].div(x.sort_values(["ticker","date"]).groupby("ticker")[OHLC].shift())
    coherent=(ratios.max(axis=1)-ratios.min(axis=1)).abs()<.05
    x["possible_corporate_action"]=x.extreme_close_return & coherent
    bad=["invalid_nonpositive_price","invalid_nonfinite_price","invalid_high_rule","invalid_low_rule","invalid_high_low"]
    x["ohlc_valid"]=~x[bad].any(axis=1); x["target_usable"]=x.ohlc_valid.astype("int8")
    return x
def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); a=p.parse_args(); x=audit_ohlc(pd.read_parquet(a.input)); setup_logging().info("invalid=%d",int((~x.ohlc_valid).sum()))
if __name__=="__main__": main()

