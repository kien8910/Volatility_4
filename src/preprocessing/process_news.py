"""Normalize hierarchy text, masks, hashes, deduplication, and filing persistence."""
from __future__ import annotations
import argparse, re
import numpy as np
import pandas as pd
from .common import GROUPS, stable_hash, setup_logging
MISSING={"","null","none","nan"}
def normalize_text(v) -> tuple[str,int]:
    if v is None or (not isinstance(v,(list,dict)) and pd.isna(v)): return "",1
    s=re.sub(r"\s+"," ",str(v)).strip()
    return ("",1) if s.lower() in MISSING else (s,0)
def process_news(df: pd.DataFrame, separator=" [SEP] "):
    x=df.copy(); rows=[]
    for hierarchy,cols in GROUPS.items():
        normalized=[]
        for c in cols:
            pair=x[c].map(normalize_text); text=pair.str[0]; miss=pair.str[1].astype("int8"); x[c+"_missing"]=miss; normalized.append(text)
            for idx,(d,t,s,m) in enumerate(zip(x.date,x.ticker,text,miss)): rows.append((idx,d,t,hierarchy,c,s,stable_hash(s) if s else "",m))
        grouped=[]
        for values in zip(*normalized): grouped.append(separator.join(f"[CATEGORY_{i+1}] {v}" for i,v in enumerate(values) if v))
        field=hierarchy+"_text"; x[field]=grouped; x[hierarchy+"_count"]=sum((s.ne("") for s in normalized),start=pd.Series(0,index=x.index)).astype("int8"); x["has_"+hierarchy]=x[hierarchy+"_count"].gt(0).astype("int8")
        x[field+"_char_length"]=x[field].str.len(); x[field+"_word_count"]=x[field].str.split().str.len(); x[field+"_hash"]=x[field].map(lambda s:stable_hash(s) if s else "")
    long=pd.DataFrame(rows,columns=["row_id","date","ticker","hierarchy","category","text","text_hash","is_missing"])
    duplicate=long.groupby(["date","hierarchy","category","text_hash"],dropna=False).cumcount().gt(0) & long.text_hash.ne("")
    long["is_duplicate_within_date"]=duplicate.astype("int8")
    # Combined filing context version/change, ordered per ticker.
    x=x.sort_values(["ticker","date"]); x["filing_version_hash"]=x.filing_text_hash
    prev=x.groupby("ticker").filing_version_hash.shift(); x["filing_changed"]=(x.filing_version_hash.ne(prev)&x.filing_version_hash.ne("")).astype("int8")
    changed_date=x.date.where(x.filing_changed.eq(1)).groupby(x.ticker).ffill(); x["days_since_filing_change"]=(x.date-changed_date).dt.days.astype("Float64")
    return x,long
def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); a=p.parse_args(); x,l=process_news(pd.read_parquet(a.input)); setup_logging().info("panel=%d long=%d",len(x),len(l))
if __name__=="__main__": main()
