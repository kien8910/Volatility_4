"""Generate conservative text leakage candidates; never auto-adjudicate them."""
from __future__ import annotations
import argparse,re
import pandas as pd
PHRASES=re.compile(r"\b(?:shares rose|shares fell|closed higher|closed lower|after reporting|following the announcement)\b",re.I)
DATE_RE=r"\b(20\d{2}[-/][01]?\d[-/][0-3]?\d)\b"
def leakage_candidates(long: pd.DataFrame)->pd.DataFrame:
    z=long.loc[long.is_missing.eq(0)].copy(); z["possible_contemporaneous_information"]=z.text.str.contains(PHRASES,na=False)
    mentioned=pd.to_datetime(z.text.str.extract(DATE_RE,expand=False).str.replace("/","-",regex=False),errors="coerce")
    z["possible_future_reference"]=mentioned.gt(z.date)
    z["event_date_mentioned"]=mentioned.dt.strftime("%Y-%m-%d").fillna("")
    return z.loc[z.possible_future_reference|z.possible_contemporaneous_information]
def assert_target_order(panel:pd.DataFrame,horizons=(1,5,10,22)):
    for h in horizons:
        c=f"target_date_h{h}"; bad=panel[c].notna()&panel[c].le(panel.date)
        if bad.any(): raise ValueError(f"Leakage: {int(bad.sum())} rows have {c} <= feature date")
def main():
    p=argparse.ArgumentParser();p.add_argument("news_long");a=p.parse_args();print(len(leakage_candidates(pd.read_parquet(a.news_long))))
if __name__=="__main__":main()
