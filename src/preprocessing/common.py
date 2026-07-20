"""Shared constants, I/O, logging, hashing, and provenance helpers."""
from __future__ import annotations
import hashlib, json, logging, platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

KEY = ["date", "ticker"]
OHLC = ["open", "high", "low", "close"]
GROUPS = {
 "macro": [f"macro_category{i}" for i in range(1, 6)],
 "sector": [f"sector_category{i}" for i in range(1, 6)],
 "target_company": [f"targetCompany_category{i}" for i in range(1, 4)],
 "related_company": [f"relatedCompany_category{i}" for i in range(1, 4)],
 "filing": ["filing_financialStatement", "filing_governanceRisks", "filing_overviewProduct", "filing_recentEventCatalyst", "filing_strategyMarketOps"],
}
REQUIRED = KEY + OHLC + sum(GROUPS.values(), [])

def setup_logging(path: Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path:
        path.parent.mkdir(parents=True, exist_ok=True); handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    return logging.getLogger("fintexts.step0")

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f: return yaml.safe_load(f)

def ensure_dirs(root: Path) -> None:
    for p in ["data/raw","data/interim","data/processed","data/quarantine","data/splits","results/tables","results/figures","results/logs","reports","experiments/step_0_data"]: (root/p).mkdir(parents=True, exist_ok=True)

def stable_hash(value: str) -> str: return hashlib.sha256(value.encode("utf-8")).hexdigest()
def dataframe_hash(df: pd.DataFrame) -> str: return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
def utc_now() -> str: return datetime.now(timezone.utc).isoformat()

def versions() -> dict[str, str]:
    import datasets, scipy, statsmodels
    return {"python": platform.python_version(), "datasets": datasets.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

