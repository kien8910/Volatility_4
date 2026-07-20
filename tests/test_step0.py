import numpy as np
import pandas as pd
import pytest
from src.preprocessing.common import REQUIRED, dataframe_hash
from src.preprocessing.validate_schema import validate_schema
from src.preprocessing.audit_panel import resolve_duplicates
from src.preprocessing.clean_ohlc import audit_ohlc
from src.preprocessing.build_volatility_targets import build_targets
from src.preprocessing.process_news import normalize_text,process_news
from src.preprocessing.create_splits import create_splits
from src.preprocessing.audit_leakage import assert_target_order

def frame(n=30):
    dates=pd.bdate_range("2020-01-01",periods=n); d=pd.DataFrame({"date":dates,"ticker":"AAA","open":100.,"high":102.,"low":99.,"close":101.})
    for c in REQUIRED:
        if c not in d: d[c]=""
    return validate_schema(d)

def test_required_schema():
    with pytest.raises(ValueError,match="missing required"): validate_schema(pd.DataFrame({"date":[]}))
def test_unique_key_and_duplicate_types():
    d=frame(2); z=pd.concat([d,d.iloc[[0]]]);clean,conflict,exact=resolve_duplicates(z);assert exact==1 and not len(conflict) and not clean.duplicated(["date","ticker"]).any()
    z=pd.concat([d,d.iloc[[0]].assign(close=9)]);clean,conflict,exact=resolve_duplicates(z);assert len(conflict)==2
def test_ohlc_inequalities():
    d=frame(2);d.loc[0,"high"]=98;x=audit_ohlc(d);assert not x.loc[0,"ohlc_valid"] and x.loc[0,"invalid_high_rule"]
def test_gk_rs_and_return_formula():
    x=build_targets(audit_ohlc(frame(3))); r=np.log(101/101);assert x.iloc[1].log_return==pytest.approx(r)
    row=x.iloc[0];gk=.5*np.log(102/99)**2-(2*np.log(2)-1)*np.log(101/100)**2;rs=np.log(102/101)*np.log(102/100)+np.log(99/101)*np.log(99/100)
    assert row.gk_variance_raw==pytest.approx(gk);assert row.rs_variance_raw==pytest.approx(rs)
def test_invalid_and_nonpositive_variance():
    d=frame(2);d.loc[0,["open","high","low","close"]]=100.;x=build_targets(audit_ohlc(d));assert x.loc[0,"gk_nonpositive_flag"] and x.loc[0,"gk_variance"]==1e-12
    d.loc[1,"close"]=-1;assert not audit_ohlc(d).loc[1,"ohlc_valid"]
def test_no_return_across_missing_observation():
    full=frame(3).assign(ticker="BBB");d=pd.concat([frame(3).drop(index=1),full],ignore_index=True);x=build_targets(audit_ohlc(d));aaa=x[x.ticker=="AAA"];assert pd.isna(aaa.iloc[1].log_return)
def test_missing_text_grouping_hash_and_dedup():
    assert normalize_text(" None ")== ("",1);d=frame(2);d["macro_category1"]=" hello  world ";x,l=process_news(d);assert x.macro_count.eq(1).all() and "[CATEGORY_1]" in x.iloc[0].macro_text
    q=l[(l.category=="macro_category1")];assert q.iloc[1].is_duplicate_within_date==0 # different dates
    q=pd.concat([d,d.assign(ticker="BBB")]);_,ll=process_news(q);m=ll[(ll.category=="macro_category1")&(ll.date==d.date.iloc[0])];assert m.is_duplicate_within_date.sum()==1
def test_filing_change_detection():
    d=frame(3);d["filing_financialStatement"]=["A","A","B"];x,_=process_news(d);assert x.filing_changed.tolist()==[1,0,1] and x.filing_version_hash.nunique()==2
def test_target_shift_and_order():
    x=build_targets(audit_ohlc(frame(30)));assert x.iloc[0].target_date_h5==x.iloc[5].date;assert_target_order(x)
def test_temporal_split_and_no_overlap():
    m,f=create_splits(pd.bdate_range("2015-01-01",periods=1300));test=set(m[m.is_locked_test==1].date)
    assert len(test)==252 and m.date.is_unique
    for _,g in f.groupby("fold_id"):
        tr=set(g[g.role=="train"].date);va=set(g[g.role=="validation"].date);assert not tr&va and not (tr|va)&test and max(tr)<min(va)<min(test)
def test_deterministic_sampling_and_hash():
    d=frame(20);a=d.sample(8,random_state=42);b=d.sample(8,random_state=42);assert a.index.tolist()==b.index.tolist() and dataframe_hash(a)==dataframe_hash(b)
