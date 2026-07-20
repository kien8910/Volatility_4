"""CLI orchestrator for the reproducible FinTexTS Step-0 pipeline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ensure_dirs,load_config,setup_logging,utc_now,dataframe_hash
from .download_fintexts import download
from .validate_schema import validate_schema
from .audit_panel import resolve_duplicates,panel_tables
from .clean_ohlc import audit_ohlc
from .build_volatility_targets import build_targets
from .process_news import process_news
from .audit_leakage import leakage_candidates,assert_target_order
from .create_splits import create_splits
from .generate_reports import generate

def run(config_path:str,root:Path,force=False):
    cfg=load_config(config_path);ensure_dirs(root);log=setup_logging(root/"results/logs/step0.log");raw=download(cfg,root,force);canonical=validate_schema(raw)
    clean,conflicts,exact=resolve_duplicates(canonical);conflicts.to_parquet(root/"data/quarantine/conflicting_duplicates.parquet",index=False)
    canonical.to_parquet(root/"data/interim/fintexts_canonical.parquet",index=False)
    audited=audit_ohlc(clean,cfg["audit"]["extreme_return_threshold"]);invalid=audited.loc[~audited.ohlc_valid];invalid.to_parquet(root/"data/quarantine/invalid_ohlc.parquet",index=False)
    # Invalid OHLC is explicitly excluded from modeling; raw/canonical and quarantine retain it.
    model=audited.loc[audited.ohlc_valid].copy(); model=build_targets(model,cfg["targets"]["epsilon"],cfg["targets"]["horizons"]); model,long=process_news(model,cfg["news"]["concatenate_separator"]);assert_target_order(model,cfg["targets"]["horizons"])
    candidates=leakage_candidates(long);candidates.to_parquet(root/"data/quarantine/leakage_candidates.parquet",index=False);long.to_parquet(root/"data/processed/fintexts_news_long.parquet",index=False)
    manifest,folds=create_splits(model.date,cfg["splits"]["locked_test_days"],cfg["splits"]["validation_days"],cfg["splits"]["expanding_folds"],cfg["splits"]["minimum_initial_train_days"]);manifest.to_csv(root/"data/splits/split_manifest.csv",index=False);folds.to_csv(root/"data/splits/expanding_folds.csv",index=False);model=model.merge(manifest,on="date",how="left",validate="many_to_one");model.to_parquet(root/"data/processed/fintexts_step0_panel.parquet",index=False)
    bt,bd=panel_tables(model);bt.to_csv(root/"results/tables/panel_completeness_by_ticker.csv",index=False);bd.to_csv(root/"results/tables/panel_completeness_by_date.csv",index=False)
    audited.loc[audited.possible_corporate_action].to_csv(root/"results/tables/possible_corporate_actions.csv",index=False);invalid.to_csv(root/"results/tables/invalid_ohlc_rows.csv",index=False)
    pd.DataFrame([{"metric":"invalid_rows","value":len(invalid)},{"metric":"possible_corporate_actions","value":int(audited.possible_corporate_action.sum())}]).to_csv(root/"results/tables/ohlc_audit_summary.csv",index=False)
    # Deterministic stratified candidate sample; reviewer fields intentionally blank.
    pool=long[long.is_missing.eq(0)].copy(); n=min(cfg["audit"]["manual_news_sample_size"],len(pool));sample=pool.groupby("hierarchy",group_keys=False).apply(lambda z:z.sample(min(len(z),max(1,n//5)),random_state=cfg["reproducibility"]["seed"]),include_groups=False).head(n).reset_index(drop=True)
    sample["possible_future_reference"]=False;sample["possible_contemporaneous_information"]=False;sample["event_date_mentioned"]="";sample["reviewer_decision"]="";sample["reviewer_note"]="";sample.to_csv(root/"reports/manual_news_audit_sample.csv",index=False)
    stats={"raw_rows":len(raw),"invalid_ohlc":len(invalid),"corp":int(audited.possible_corporate_action.sum()),"gk_nonpos":int(model.gk_nonpositive_flag.sum()),"rs_nonpos":int(model.rs_nonpositive_flag.sum()),"hard_ok":not len(conflicts) and not model.duplicated(["date","ticker"]).any()}
    decision,cov,persist=generate(model,long,manifest,folds,root,stats)
    logrows=[{"timestamp":utc_now(),"step":"duplicates","rule":"exact duplicate","affected_rows":exact,"affected_tickers":0,"action":"deduplicate","output_file":"canonical/modeling","note":"Exact copies only"},{"timestamp":utc_now(),"step":"duplicates","rule":"conflicting key","affected_rows":len(conflicts),"affected_tickers":conflicts.ticker.nunique() if len(conflicts) else 0,"action":"quarantine","output_file":"data/quarantine/conflicting_duplicates.parquet","note":"No arbitrary selection"},{"timestamp":utc_now(),"step":"ohlc","rule":"invalid OHLC","affected_rows":len(invalid),"affected_tickers":invalid.ticker.nunique() if len(invalid) else 0,"action":"quarantine and exclude","output_file":"data/quarantine/invalid_ohlc.parquet","note":"No interpolation/fill"}];pd.DataFrame(logrows).to_csv(root/"results/logs/data_cleaning_log.csv",index=False)
    (root/"reports/data_dictionary.md").write_text("# Data Dictionary\n\n`date` is row/feature date; `target_date_h*` is the future observed row date. OHLC are float64 and available after close. `log_return=ln(C_t/C_{t-1})`; GK and Rogers–Satchell raw variances preserve estimator output, clipped variants use configured epsilon, and `logvol=0.5 ln(variance)`. Text fields preserve hierarchy; missing text is empty with category masks. Filing is persistent context, not daily news. Targets are never features. News publication time is unknown, so same-day use risks leakage.\n",encoding="utf-8")
    summary={**stats,"processed_rows":len(model),"tickers":model.ticker.nunique(),"dates":model.date.nunique(),"exact_duplicates":exact,"conflicting_duplicate_rows":len(conflicts),"leakage_candidates":len(candidates),"output_hash":dataframe_hash(model),"decision":decision,"news":cov.to_dict("records")};(root/"results/step0_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8");log.info("DECISION: %s",decision);return summary
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/data.yaml");p.add_argument("--root",default=".");p.add_argument("--force-download",action="store_true");a=p.parse_args();print(run(a.config,Path(a.root).resolve(),a.force_download))
if __name__=="__main__":main()

