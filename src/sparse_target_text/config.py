from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    required = ["data", "information_cutoff", "event_filter", "text_encoder", "model", "training", "output"]
    missing = [section for section in required if section not in cfg]
    if missing:
        raise ValueError(f"Sparse target-text config missing sections: {missing}")
    horizons = tuple(int(x) for x in cfg["data"]["horizons"])
    if any(h not in {1, 5} for h in horizons):
        raise ValueError("This pilot intentionally supports only horizons 1 and 5.")
    if int(cfg["information_cutoff"].get("news_lag_sessions", 1)) < 0:
        raise ValueError("news_lag_sessions must be non-negative")
    return cfg
