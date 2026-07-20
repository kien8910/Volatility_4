"""Download an immutable, reproducible FinTexTS parquet snapshot."""
from __future__ import annotations
import argparse, os, glob
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from .common import dataframe_hash, load_config, setup_logging, utc_now, versions, write_json

def download(config: dict, root: Path, force: bool = False):
    log = setup_logging(); raw = root/"data/raw/fintexts_train.parquet"; meta = root/"data/raw/provenance.json"
    if raw.exists() and not force:
        import pandas as pd
        log.info("Using existing snapshot %s", raw); return pd.read_parquet(raw)
    spec = config["dataset"]
    # In offline/restricted environments, use the exact HF Arrow cache produced by
    # load_dataset rather than waiting on Hub metadata. This is still the immutable
    # HF snapshot, not an external substitute.
    cached=glob.glob(str(Path.home()/".cache/huggingface/datasets/EXAONE-BI___fin_tex_ts/default/*/*/fin_tex_ts-train-*.arrow"))
    if os.environ.get("HF_DATASETS_OFFLINE") and cached:
        ds=concatenate_datasets([Dataset.from_file(f) for f in sorted(cached)])
    else:
        ds = load_dataset(spec["id"], split=spec["split"], revision=spec.get("revision"))
    df = ds.to_pandas(); df.to_parquet(raw, index=False)
    try: os.chmod(raw, 0o444)
    except OSError: log.warning("Could not mark snapshot read-only")
    info = {"dataset_id": spec["id"], "split": spec["split"], "requested_revision": spec.get("revision"), "fingerprint": getattr(ds, "_fingerprint", None), "downloaded_at_utc": utc_now(), "rows": len(df), "columns": len(df.columns), "bytes": raw.stat().st_size, "sha256_dataframe": dataframe_hash(df), "versions": versions(), "seed": config["reproducibility"]["seed"]}
    write_json(info, meta); log.info("Saved %d rows", len(df)); return df

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/data.yaml"); p.add_argument("--root",default="."); p.add_argument("--force",action="store_true"); a=p.parse_args()
    download(load_config(a.config),Path(a.root),a.force)
if __name__ == "__main__": main()
