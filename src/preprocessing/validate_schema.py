"""Validate required schema and canonicalize safe scalar types."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .common import REQUIRED, OHLC, setup_logging

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing=sorted(set(REQUIRED)-set(df.columns))
    if missing: raise ValueError("FinTexTS schema validation failed; missing required columns: "+", ".join(missing))
    out=df.copy(); out["date"]=pd.to_datetime(out["date"],errors="coerce").dt.normalize()
    if out["date"].isna().any(): raise ValueError(f"Invalid dates: {int(out['date'].isna().sum())}")
    out["ticker"]=out["ticker"].astype("string").str.strip()
    if out["ticker"].isna().any() or out["ticker"].eq("").any(): raise ValueError("Missing/blank ticker values")
    for c in OHLC: out[c]=pd.to_numeric(out[c],errors="coerce").astype("float64")
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); a=p.parse_args(); df=validate_schema(pd.read_parquet(a.input)); setup_logging().info("Schema OK: %d x %d",*df.shape)
if __name__=="__main__": main()

