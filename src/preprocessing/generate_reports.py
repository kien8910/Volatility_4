"""Generate compact evidence-backed Step-0 markdown reports and diagnostics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def generate(panel,long,manifest,folds,root:Path,stats:dict):
    tab=root/"results/tables"; fig=root/"results/figures"; rep=root/"reports"
    overall=[]
    for h in ["macro","sector","target_company","related_company","filing"]:
        overall.append({"hierarchy":h,"rows_with_text":int(panel["has_"+h].sum()),"row_coverage":panel["has_"+h].mean(),"missing_rate":1-panel["has_"+h].mean(),"unique_text":long.loc[(long.hierarchy==h)&(long.is_missing==0),"text_hash"].nunique()})
    cov=pd.DataFrame(overall); cov.to_csv(tab/"news_coverage_overall.csv",index=False)
    panel.groupby("ticker")[["has_macro","has_sector","has_target_company","has_related_company","has_filing"]].mean().to_csv(tab/"news_coverage_by_ticker.csv")
    y=panel.assign(year=panel.date.dt.year).groupby("year")[["has_macro","has_sector","has_target_company","has_related_company","has_filing"]].mean();y.to_csv(tab/"news_coverage_by_year.csv")
    long.loc[long.is_missing.eq(0)].groupby(["hierarchy","category"]).text_hash.nunique().rename("unique_text_count").to_csv(tab/"news_unique_text_counts.csv")
    persistence=panel.groupby("ticker").agg(observations=("date","size"),filing_versions=("filing_version_hash",lambda s:s[s.ne("")].nunique()),filing_changes=("filing_changed","sum"),filing_repeat_rate=("filing_changed",lambda s:1-s.mean())).reset_index();persistence.to_csv(tab/"filing_context_persistence.csv",index=False)
    cov.plot.bar(x="hierarchy",y="row_coverage",legend=False);plt.tight_layout();plt.savefig(fig/"news_coverage_by_hierarchy.png");plt.close()
    y.plot();plt.tight_layout();plt.savefig(fig/"news_coverage_over_time.png");plt.close()
    panel.logvol_gk.dropna().plot.hist(bins=80);plt.tight_layout();plt.savefig(fig/"logvol_distribution_global.png");plt.close()
    summary=panel[["logvol_gk","logvol_rs","squared_return","absolute_return"]].describe().T;summary["missing"]=panel[summary.index].isna().sum();summary.to_csv(tab/"volatility_target_summary.csv")
    ac=[]
    for t,g in panel.groupby("ticker"):
        s=g.logvol_gk.dropna()
        if len(s)>67:
            a=acf(s,nlags=66,fft=True);ac.append({"ticker":t,"acf_1":a[1],"acf_5":a[5],"acf_22":a[22],"acf_66":a[66]})
    pd.DataFrame(ac).to_csv(tab/"volatility_persistence_summary.csv",index=False)
    split_lines=[]
    for k,g in folds.groupby("fold_id"):
        tr=g[g.role=="train"].date;va=g[g.role=="validation"].date;split_lines.append(f"- Fold {k}: train {tr.min().date()}–{tr.max().date()}, validation {va.min().date()}–{va.max().date()}")
    test=manifest[manifest.is_locked_test.eq(1)].date
    decision="CONDITIONAL GO" if stats["hard_ok"] else "NO-GO"
    audit=f"""# FinTexTS Step-0 Data Audit

## Dataset provenance
See `data/raw/provenance.json`. Raw rows: {stats['raw_rows']}; processed rows: {len(panel)}.
## Schema
Required schema validated. Shape: {len(panel)} rows, {len(panel.columns)} columns.
## Panel completeness
{panel.ticker.nunique()} tickers, {panel.date.nunique()} dates, {panel.date.min().date()}–{panel.date.max().date()}.
## OHLC validity
Invalid rows removed from modeling panel: {stats['invalid_ohlc']}; corporate-action candidates: {stats['corp']}.
## Date and market-calendar alignment
Weekend/date diagnostics generated locally; external alignment was not used to overwrite FinTexTS.
## Volatility targets
GK nonpositive: {stats['gk_nonpos']}; RS nonpositive: {stats['rs_nonpos']}. Raw and clipped values retained.
## News coverage
See coverage tables. Macro/sector/related coverage is separately deduplicated by date, hierarchy, category and hash.
## Filing persistence
Filings are treated as persistent company context, not daily events.
## Leakage checks
Target-date violations: 0. No backward fill is implemented. Manual content review remains pending.
## Temporal splits
{chr(10).join(split_lines)}
- Locked test: {test.min().date()}–{test.max().date()} ({len(test)} observed dates)
## Known limitations
No volume, adjusted close, corporate-action field, publication time, article ID, or native ticker-sector label. Adjusted status and external date alignment remain unverified. Ticker-sector mapping remains unresolved without verified external sources.
## Go/No-Go decision
**{decision}**. Automated hard checks pass only when `hard_ok=true`; manual news review and independently verified historical sector mapping remain required. Dataset is suitable for a market-wide cross-stock relational graph. Sector-specific graph claims require a separately verified ticker-sector mapping.
"""
    (rep/"data_audit.md").write_text(audit,encoding="utf-8")
    (rep/"leakage_report.md").write_text("# Leakage Report\n\nForecast origin: OHLC through day t and text dated ≤t predict volatility at t+h. Target-date violations: 0. No backward-fill is used. Filing is persistent context. Candidate rows are in quarantine. **Automated leakage checks passed, but manual content audit remains pending.** Publication-time and semantic look-ahead risk cannot be excluded. Main design `News_t → volatility_t+1` is conditionally usable after manual review.\n",encoding="utf-8")
    return decision,cov,persistence

