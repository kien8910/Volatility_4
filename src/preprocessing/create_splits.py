"""Create locked-test and four non-overlapping expanding validation manifests."""
from __future__ import annotations
import argparse
import pandas as pd
from .common import setup_logging
def create_splits(dates, locked_test_days=252, validation_days=126, folds=4, minimum_train=504):
    d=pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(dates).unique()))); n=len(d)
    if n < locked_test_days+minimum_train+folds: raise ValueError(f"Only {n} dates; need at least {locked_test_days+minimum_train+folds}")
    dev=d[:-locked_test_days]; test=d[-locked_test_days:]
    val=min(validation_days,(len(dev)-minimum_train)//folds)
    if val<1: raise ValueError("Insufficient dates for expanding folds")
    val_start=len(dev)-folds*val; records=[]
    for k in range(folds):
        vs=val_start+k*val; ve=vs+val
        records += [(k+1,z,"train") for z in dev[:vs]]+[(k+1,z,"validation") for z in dev[vs:ve]]
    manifest=pd.DataFrame({"date":d,"base_split":["development"]*len(dev)+["locked_test"]*len(test),"is_locked_test":[0]*len(dev)+[1]*len(test)})
    return manifest,pd.DataFrame(records,columns=["fold_id","date","role"])
def main():
    p=argparse.ArgumentParser();p.add_argument("input");a=p.parse_args();x=pd.read_parquet(a.input);m,f=create_splits(x.date);setup_logging().info("manifest=%d folds=%d",len(m),len(f))
if __name__=="__main__":main()

