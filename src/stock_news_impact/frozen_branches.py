from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def choose_step6_news_branch(cfg: dict) -> dict:
    out_dir = Path(cfg["experiment"]["output_dir"])
    configured = Path(cfg["data"].get("step6_config_path", ""))
    if configured.exists():
        doc = yaml.safe_load(configured.read_text(encoding="utf-8"))
        selected = {
            "selection_rule": "read from Step 6 best_naive_news_config.yaml",
            "model": str(doc.get("model", "")),
            "config_id": str(doc.get("config_id", "")),
            "validation_qlike": float(doc.get("validation_qlike", "nan")),
        }
    else:
        pred_path = Path(cfg["data"]["step6_predictions_path"])
        pred = pd.read_parquet(pred_path)
        metrics = pred.groupby(["model", "config_id"], as_index=False)["qlike_loss"].mean()
        news = metrics.loc[~metrics["model"].astype(str).eq("stock_only")].copy()
        if news.empty:
            selected = {
                "selection_rule": "no Step 6 news branch available; use zero correction proxy",
                "model": "none",
                "config_id": "none",
                "validation_qlike": float("nan"),
            }
        else:
            best = news.sort_values("qlike_loss").iloc[0]
            selected = {
                "selection_rule": "least harmful Step 6 news branch on validation; locked test not used",
                "model": str(best["model"]),
                "config_id": str(best["config_id"]),
                "validation_qlike": float(best["qlike_loss"]),
            }
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(selected, (out_dir / "selected_news_branch.yaml").open("w", encoding="utf-8"), sort_keys=False)
    return selected


def news_correction_proxy(step6_predictions: pd.DataFrame, selected_branch: dict) -> pd.DataFrame:
    config_id = str(selected_branch.get("config_id", "none"))
    if config_id == "none":
        stock = step6_predictions.loc[step6_predictions["model"].astype(str).eq("stock_only")].copy()
        stock["news_correction_proxy"] = 0.0
        return stock[["date", "target_date", "ticker", "horizon", "fold_id", "seed", "news_correction_proxy"]]
    sub = step6_predictions.loc[step6_predictions["config_id"].astype(str).eq(config_id)].copy()
    if sub.empty:
        raise ValueError(f"Selected Step 6 branch has no prediction rows: {config_id}")
    sub["news_correction_proxy"] = sub["news_residual_correction"].astype(float)
    return sub[["date", "target_date", "ticker", "horizon", "fold_id", "seed", "news_correction_proxy"]]
