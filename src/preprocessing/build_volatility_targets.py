"""Compute float64 return, range estimators, and future-row targets."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .common import setup_logging

def build_targets(df: pd.DataFrame, epsilon: float=1e-12, horizons=(1,5,10,22)) -> pd.DataFrame:
    x=df.sort_values(["ticker","date"]).copy(); g=x.groupby("ticker",sort=False)
    prev_close=g.close.shift(); prev_date=g.date.shift()
    # Adjacency is defined on the panel's observed-date index, not calendar-day distance.
    # This permits weekends/holidays but prevents returns across a missing panel row.
    date_rank={d:i for i,d in enumerate(sorted(x.date.unique()))}; rank=x.date.map(date_rank); prev_rank=prev_date.map(date_rank)
    consecutive=rank.sub(prev_rank).eq(1) & x.ohlc_valid & g.ohlc_valid.shift().astype("boolean").fillna(False)
    x["log_return"]=np.where(consecutive,np.log(x.close/prev_close),np.nan).astype("float64")
    x["squared_return"]=x.log_return.pow(2); x["absolute_return"]=x.log_return.abs()
    valid=x.ohlc_valid
    hl=np.log(x.high/x.low); co=np.log(x.close/x.open)
    x["gk_variance_raw"]=np.where(valid,.5*hl.pow(2)-(2*np.log(2)-1)*co.pow(2),np.nan)
    x["gk_nonpositive_flag"]=x.gk_variance_raw.le(0) & x.gk_variance_raw.notna(); x["gk_variance"]=x.gk_variance_raw.clip(lower=epsilon); x["logvol_gk"]=.5*np.log(x.gk_variance)
    hc=np.log(x.high/x.close); ho=np.log(x.high/x.open); lc=np.log(x.low/x.close); lo=np.log(x.low/x.open)
    x["rs_variance_raw"]=np.where(valid,hc*ho+lc*lo,np.nan)
    x["rs_nonpositive_flag"]=x.rs_variance_raw.le(0) & x.rs_variance_raw.notna(); x["rs_variance"]=x.rs_variance_raw.clip(lower=epsilon); x["logvol_rs"]=.5*np.log(x.rs_variance)
    for h in horizons:
        x[f"target_date_h{h}"]=g.date.shift(-h)
        for col in ["logvol_gk","logvol_rs","squared_return","absolute_return"]: x[f"target_{col}_h{h}"]=g[col].shift(-h)
    return x
def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--epsilon",type=float,default=1e-12); a=p.parse_args(); x=build_targets(pd.read_parquet(a.input),a.epsilon); setup_logging().info("targets rows=%d",len(x))
if __name__=="__main__": main()
