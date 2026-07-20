"""Resolve exact duplicates, quarantine conflicts, and report panel coverage."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .common import KEY, OHLC, setup_logging

def resolve_duplicates(df: pd.DataFrame):
    exact=df.duplicated(keep="first"); work=df.loc[~exact].copy()
    dup=work.duplicated(KEY,keep=False); conflicts=work.loc[dup].copy(); clean=work.loc[~dup].copy()
    return clean, conflicts, int(exact.sum())

def panel_tables(df: pd.DataFrame):
    dates=pd.Index(sorted(df.date.unique())); n=len(dates)
    by_t=(df.groupby("ticker").agg(first_date=("date","min"),last_date=("date","max"),observations=("date","nunique")).reset_index())
    by_t["missing_dates"]=n-by_t.observations
    by_d=df.groupby("date").ticker.nunique().rename("ticker_count").reset_index()
    return by_t,by_d

def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); a=p.parse_args(); df=pd.read_parquet(a.input); c,q,e=resolve_duplicates(df); setup_logging().info("exact=%d conflicts=%d clean=%d",e,len(q),len(c))
if __name__=="__main__": main()

