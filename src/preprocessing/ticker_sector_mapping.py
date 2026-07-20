"""Validate a separately sourced historical ticker-sector map and create masks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
REQUIRED=["ticker","company_name","sector_code","sector_name","industry_code","industry_name","classification_system","source","effective_date","retrieval_date","mapping_confidence","mapping_note"]
USER_COLUMNS=["ticker","company_name","sector_name","sector_code","industry_name","industry_code","yahoo_industry"]
SECTOR_ALIASES={
    "Technology":("45","Information Technology"),
    "Healthcare":("35","Health Care"),
    "Financial Services":("40","Financials"),
    "Consumer Defensive":("30","Consumer Staples"),
    "Consumer Cyclical":("25","Consumer Discretionary"),
    "Basic Materials":("15","Materials"),
}
SECTOR_CODES={
    "Energy":"10",
    "Materials":"15",
    "Industrials":"20",
    "Consumer Discretionary":"25",
    "Consumer Staples":"30",
    "Health Care":"35",
    "Financials":"40",
    "Information Technology":"45",
    "Communication Services":"50",
    "Utilities":"55",
    "Real Estate":"60",
}

def _clean_code(value):
    if pd.isna(value) or value == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()

def load_user_mapping(mapping_path:Path)->pd.DataFrame:
    """Load a user-maintained ticker/GICS workbook or CSV into the required schema."""
    if mapping_path.suffix.lower() in {".xlsx",".xls"}:
        user=pd.read_excel(mapping_path)
    else:
        user=pd.read_csv(mapping_path)
    if len(user.columns) < 6:
        raise ValueError("Ticker-sector mapping needs at least ticker, company, sector, sector code, industry, industry code columns.")
    user=user.iloc[:, : min(len(user.columns), len(USER_COLUMNS))].copy()
    user.columns=USER_COLUMNS[: len(user.columns)]
    if "yahoo_industry" not in user:
        user["yahoo_industry"]=""
    user["ticker"]=user["ticker"].astype(str).str.strip().str.upper()
    user=user[user["ticker"].ne("") & user["ticker"].ne("NAN")].copy()

    sector_name=user["sector_name"].fillna("").astype(str).str.strip()
    alias=sector_name.map(SECTOR_ALIASES)
    alias_name=alias.map(lambda x:x[1] if isinstance(x,tuple) else "")
    user["sector_name"]=sector_name.where(alias_name.eq(""), alias_name)
    user["sector_code"]=user["sector_code"].map(_clean_code)
    alias_code=alias.map(lambda x:x[0] if isinstance(x,tuple) else "")
    user.loc[user["sector_code"].eq("") & alias_code.ne(""),"sector_code"]=alias_code
    user.loc[user["sector_code"].eq("") & user["sector_name"].isin(SECTOR_CODES),"sector_code"]=user.loc[user["sector_code"].eq("") & user["sector_name"].isin(SECTOR_CODES),"sector_name"].map(SECTOR_CODES)
    user["industry_code"]=user["industry_code"].map(_clean_code)
    user["industry_name"]=user["industry_name"].fillna("").astype(str).str.strip()
    user["company_name"]=user["company_name"].fillna("").astype(str).str.strip()
    today=pd.Timestamp.now().date().isoformat()
    out=pd.DataFrame({
        "ticker":user["ticker"],
        "company_name":user["company_name"],
        "sector_code":user["sector_code"],
        "sector_name":user["sector_name"],
        "industry_code":user["industry_code"],
        "industry_name":user["industry_name"],
        "classification_system":"GICS/user_supplied",
        "source":str(mapping_path),
        "effective_date":"",
        "retrieval_date":today,
        "mapping_confidence":"probable",
        "mapping_note":user["yahoo_industry"].fillna("").astype(str).str.strip(),
    })
    unresolved=out.sector_code.eq("") | out.sector_name.eq("")
    out.loc[unresolved,"mapping_confidence"]="unresolved"
    return out

def build_masks(mapping:pd.DataFrame,tickers:list[str]):
    miss=set(REQUIRED)-set(mapping); 
    if miss: raise ValueError("Sector mapping missing: "+", ".join(sorted(miss)))
    m=mapping.drop_duplicates("ticker").set_index("ticker").reindex(tickers); verified=m.mapping_confidence.isin(["verified","probable"]).to_numpy()
    sector_code=m.sector_code.fillna("").astype(str).to_numpy(); industry_code=m.industry_code.fillna("").astype(str).to_numpy()
    sector_valid=verified & (sector_code != ""); industry_valid=verified & (industry_code != "")
    sec=np.equal.outer(sector_code,sector_code)&np.outer(sector_valid,sector_valid); ind=np.equal.outer(industry_code,industry_code)&np.outer(industry_valid,industry_valid)
    np.fill_diagonal(sec,False);np.fill_diagonal(ind,False);cross=(~sec)&np.outer(sector_valid,sector_valid);np.fill_diagonal(cross,False)
    return sec.astype("int8"),ind.astype("int8"),cross.astype("int8")

def write_group_tables(panel:pd.DataFrame,mapping:pd.DataFrame,root:Path):
    tickers=sorted(panel.ticker.unique())
    aligned=mapping.drop_duplicates("ticker").set_index("ticker").reindex(tickers).reset_index()
    g=panel.groupby("ticker").agg(first_date=("date","min"),last_date=("date","max"),observations=("date","size"),news_coverage_macro=("has_macro","mean"),news_coverage_sector=("has_sector","mean"),news_coverage_company=("has_target_company","mean")).reset_index()
    joined=aligned.merge(g,on="ticker",how="left")
    usable=joined.mapping_confidence.isin(["verified","probable"])
    joined.loc[~usable,"sector_name"]="UNRESOLVED"
    joined.loc[~usable,"industry_name"]="UNRESOLVED"
    def summary(level):
        return joined.groupby(level,dropna=False).agg(
            number_of_tickers=("ticker","size"),
            number_of_usable_tickers=("mapping_confidence",lambda s:int(s.isin(["verified","probable"]).sum())),
            first_date=("first_date","min"),
            last_date=("last_date","max"),
            median_number_of_observations=("observations","median"),
            news_coverage_macro=("news_coverage_macro","mean"),
            news_coverage_sector=("news_coverage_sector","mean"),
            news_coverage_company=("news_coverage_company","mean"),
        ).reset_index()
    sector_summary=summary("sector_name")
    sector_summary["eligible_for_sector_graph"]=sector_summary.number_of_usable_tickers.ge(2) & sector_summary.sector_name.ne("UNRESOLVED")
    industry_summary=summary("industry_name")
    industry_summary["eligible_for_sector_graph"]=industry_summary.number_of_usable_tickers.ge(2) & industry_summary.industry_name.ne("UNRESOLVED")
    sector_summary.to_csv(root/"results/tables/sector_group_summary.csv",index=False)
    industry_summary.to_csv(root/"results/tables/industry_group_summary.csv",index=False)
    lines=["# Stock Groups by Sector",""]
    for sector,part in joined.sort_values(["sector_name","ticker"]).groupby("sector_name",dropna=False):
        lines.extend([f"## {sector}",""])
        for _,row in part.iterrows():
            note=row.mapping_confidence if pd.notna(row.mapping_confidence) else "unresolved"
            lines.append(f"- {row.ticker} - {row.company_name if pd.notna(row.company_name) else ''} ({note})")
        lines.append("")
    (root/"reports/stock_groups_by_sector.md").write_text("\n".join(lines),encoding="utf-8")

def write_mapping_artifacts(panel_path:Path,root:Path,mapping_path:Path):
    panel=pd.read_parquet(panel_path,columns=["date","ticker","has_macro","has_sector","has_target_company"])
    tickers=sorted(panel.ticker.unique())
    supplied=load_user_mapping(mapping_path)
    aligned=supplied.drop_duplicates("ticker").set_index("ticker").reindex(tickers).reset_index()
    missing=aligned.sector_code.isna() | aligned.sector_code.eq("") | aligned.sector_name.isna() | aligned.sector_name.eq("")
    aligned.loc[missing,"mapping_confidence"]="unresolved"
    aligned.loc[aligned.mapping_confidence.isna(),"mapping_confidence"]="unresolved"
    aligned.loc[aligned.classification_system.isna(),"classification_system"]="unresolved"
    aligned.loc[aligned.source.isna(),"source"]=str(mapping_path)
    aligned.loc[aligned.retrieval_date.isna(),"retrieval_date"]=pd.Timestamp.now().date().isoformat()
    aligned.loc[aligned.mapping_note.isna(),"mapping_note"]="Ticker missing from supplied mapping or missing GICS sector fields."
    for col in REQUIRED:
        if col not in aligned:
            aligned[col]=""
    aligned[REQUIRED].to_csv(root/"data/processed/ticker_sector_mapping.csv",index=False)
    sec,ind,cross=build_masks(aligned[REQUIRED],tickers)
    out=root/"data/processed"
    np.save(out/"sector_adjacency_mask.npy",sec);np.save(out/"industry_adjacency_mask.npy",ind);np.save(out/"cross_sector_adjacency_mask.npy",cross)
    qc=pd.DataFrame({
        "check":["panel_tickers","mapping_tickers","missing_in_mapping","extra_in_mapping","unresolved_after_alignment","duplicate_mapping_rows"],
        "value":[len(tickers),supplied.ticker.nunique(),", ".join([t for t in tickers if t not in set(supplied.ticker)]),", ".join([t for t in sorted(set(supplied.ticker)-set(tickers))]),int(aligned.mapping_confidence.eq("unresolved").sum()),int(supplied.ticker.duplicated().sum())],
    })
    qc.to_csv(root/"results/tables/ticker_sector_mapping_qc.csv",index=False)
    write_group_tables(panel,aligned[REQUIRED],root)
    return qc,aligned

def write_unresolved_artifacts(panel_path:Path,root:Path):
    """Emit explicit unresolved mapping artifacts when no verified source is supplied."""
    panel=pd.read_parquet(panel_path,columns=["date","ticker","has_macro","has_sector","has_target_company"])
    tickers=sorted(panel.ticker.unique()); today=pd.Timestamp.now().date().isoformat()
    m=pd.DataFrame({"ticker":tickers,"company_name":"","sector_code":"","sector_name":"UNRESOLVED","industry_code":"","industry_name":"UNRESOLVED","classification_system":"unresolved","source":"not supplied; external verification required","effective_date":"","retrieval_date":today,"mapping_confidence":"unresolved","mapping_note":"FinTexTS has no native ticker-sector label; no inference from sector news"})
    out=root/"data/processed";m.to_csv(out/"ticker_sector_mapping.csv",index=False)
    n=len(tickers);zero=np.zeros((n,n),dtype="int8");np.save(out/"sector_adjacency_mask.npy",zero);np.save(out/"industry_adjacency_mask.npy",zero);np.save(out/"cross_sector_adjacency_mask.npy",zero)
    m.to_csv(root/"results/tables/unresolved_tickers.csv",index=False)
    g=panel.groupby("ticker").agg(first_date=("date","min"),last_date=("date","max"),observations=("date","size"),news_coverage_macro=("has_macro","mean"),news_coverage_sector=("has_sector","mean"),news_coverage_company=("has_target_company","mean"))
    summary=pd.DataFrame([{"sector_name":"UNRESOLVED","number_of_tickers":n,"number_of_usable_tickers":0,"first_date":g.first_date.min(),"last_date":g.last_date.max(),"median_number_of_observations":g.observations.median(),"news_coverage_macro":g.news_coverage_macro.mean(),"news_coverage_sector":g.news_coverage_sector.mean(),"news_coverage_company":g.news_coverage_company.mean(),"eligible_for_sector_graph":False}]);summary.to_csv(root/"results/tables/sector_group_summary.csv",index=False);summary.rename(columns={"sector_name":"industry_name"}).to_csv(root/"results/tables/industry_group_summary.csv",index=False)
    lines=["# Stock Groups by Sector","","## UNRESOLVED","",f"- Tickers: {n}","- Usable period: see per-ticker panel completeness table.","- Eligible for within-sector graph: No.","- All tickers require independently verified historical classification.",""]+[f"- {t} — company/sector not verified" for t in tickers]
    (root/"reports/stock_groups_by_sector.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--panel",default="data/processed/fintexts_step0_panel.parquet");p.add_argument("--root",default=".");p.add_argument("--mapping",default="");a=p.parse_args();root=Path(a.root).resolve()
    if a.mapping:
        write_mapping_artifacts(Path(a.panel),root,Path(a.mapping))
    else:
        write_unresolved_artifacts(Path(a.panel),root)
