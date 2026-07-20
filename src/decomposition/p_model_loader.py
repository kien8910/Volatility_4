from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _select_har_ridge_from_step1(root: Path, tickers: list[str], horizons: list[int], fallback_path: Path) -> dict[str, Any]:
    pred_path = root / "results/tables/step1_predictions_validation.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing Step-1 validation predictions: {pred_path}")
    pred = pd.read_parquet(pred_path, columns=["model", "ticker", "horizon", "hyperparams", "qlike_loss"])
    pred = pred.loc[pred["model"].eq("B4_HAR_Ridge") & pred["ticker"].isin(tickers) & pred["horizon"].isin(horizons)]
    if pred.empty:
        raise ValueError("No B4_HAR_Ridge validation predictions found for Step-3 fallback.")
    rows = []
    for (ticker, horizon), part in pred.groupby(["ticker", "horizon"], dropna=False):
        q = part.groupby("hyperparams", dropna=False)["qlike_loss"].mean().reset_index()
        best = q.sort_values(["qlike_loss", "hyperparams"]).iloc[0]
        rows.append({"ticker": ticker, "horizon": int(horizon), "hyperparams": best.hyperparams, "validation_qlike": float(best.qlike_loss)})
    selected = pd.DataFrame(rows)
    cfg = {
        "source": "step1_validation",
        "model_name": "HAR-Ridge",
        "fallback_used": True,
        "features": ["har_d", "har_w", "har_m"],
        "selection_metric": "validation_qlike",
        "selected": selected.to_dict("records"),
    }
    write_yaml(cfg, fallback_path)
    return cfg


def load_p_model_config(config: dict[str, Any], root: Path) -> dict[str, Any]:
    primary = root / config["p_model"]["primary_config"]
    fallback = root / config["p_model"]["fallback_config"]
    tickers = [str(t).upper() for t in config["data"]["tickers"]]
    horizons = [int(h) for h in config["target"]["horizons"]]
    if primary.exists():
        cfg = read_yaml(primary)
        if str(cfg.get("decision", "")).upper() in {"GO", "CONDITIONAL GO"}:
            cfg["fallback_used"] = False
            return cfg
    if fallback.exists():
        cfg = read_yaml(fallback)
        cfg["fallback_used"] = True
        return cfg
    return _select_har_ridge_from_step1(root, tickers, horizons, fallback)


def alpha_for(config: dict[str, Any], ticker: str, horizon: int) -> float:
    import json

    default_alpha = 1.0
    for row in config.get("selected", []):
        if str(row.get("ticker", "")).upper() == ticker and int(row.get("horizon", -1)) == int(horizon):
            hyper = row.get("hyperparams", {})
            if isinstance(hyper, str):
                hyper = json.loads(hyper)
            return float(hyper.get("alpha", default_alpha))
    return default_alpha
