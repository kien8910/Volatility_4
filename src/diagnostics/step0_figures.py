"""Create the remaining panel and volatility diagnostic figures."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram

def run(panel_path:Path,root:Path):
    x=pd.read_parquet(panel_path);f=root/"results/figures";t=root/"results/tables"
    # Missingness heatmap uses observed panel-key presence before any imputation.
    p=x.assign(v=1).pivot(index="ticker",columns="date",values="v").isna()
    plt.figure(figsize=(14,6));plt.imshow(p,aspect="auto",interpolation="nearest",cmap="Greys");plt.xlabel("observed date index");plt.ylabel("ticker");plt.tight_layout();plt.savefig(f/"panel_missingness_heatmap.png",dpi=150);plt.close()
    examples=sorted(x.ticker.unique())[:6]
    x[x.ticker.isin(examples)].boxplot(column="logvol_gk",by="ticker",grid=False);plt.suptitle("");plt.tight_layout();plt.savefig(f/"logvol_distribution_by_ticker.png");plt.close()
    fig,axes=plt.subplots(3,2,figsize=(12,8))
    for ax,ticker in zip(axes.flat,examples):
        s=x.loc[x.ticker==ticker,"logvol_gk"].dropna();a=acf(s,nlags=min(66,len(s)-1),fft=True);ax.stem(range(len(a)),a);ax.set_title(ticker)
    plt.tight_layout();plt.savefig(f/"logvol_acf_examples.png");plt.close()
    fig,axes=plt.subplots(3,2,figsize=(12,8))
    for ax,ticker in zip(axes.flat,examples):
        s=x.loc[x.ticker==ticker,"logvol_gk"].dropna();freq,pow=periodogram(s);ax.plot(freq[1:],pow[1:]);ax.set_title(ticker)
    plt.tight_layout();plt.savefig(f/"logvol_periodogram_examples.png");plt.close()
    fig,axes=plt.subplots(3,2,figsize=(12,8))
    for ax,ticker in zip(axes.flat,examples):
        q=x[x.ticker==ticker];ax.plot(q.date,q.logvol_gk,lw=.6);ax.set_title(ticker)
    plt.tight_layout();plt.savefig(f/"logvol_time_series_examples.png");plt.close()
    # Date diagnostics are evidence only; no NYSE calendar assumptions or deletion.
    by=x.groupby("date").agg(tickers=("ticker","nunique"),unique_close=("close","nunique")).reset_index();by["is_weekend"]=by.date.dt.dayofweek.ge(5);by.to_csv(t/"date_calendar_diagnostics.csv",index=False)
    pd.DataFrame(columns=["ticker","date","fintexts_open","fintexts_high","fintexts_low","fintexts_close","external_source","external_date","alignment_status","note"]).to_csv(t/"external_ohlc_alignment_check.csv",index=False)
    # Explicit manifests for later experiments; these contain no learned transforms.
    x[["date","ticker"]].assign(design="News_t_to_volatility_t",usage="diagnostic_only").to_csv(root/"data/splits/same_day_diagnostic_manifest.csv",index=False)
    x[["date","ticker","target_date_h1"]].assign(design="News_t_to_volatility_t_plus_1",usage="primary").to_csv(root/"data/splits/next_day_primary_manifest.csv",index=False)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--panel",default="data/processed/fintexts_step0_panel.parquet");p.add_argument("--root",default=".");a=p.parse_args();run(Path(a.panel),Path(a.root).resolve())
