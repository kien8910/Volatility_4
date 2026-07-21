from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph.metrics import qlike_from_logvol, summarize_predictions


def add_qlike(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = frame.copy()
    qlike, clipped = qlike_from_logvol(
        out["actual_logvol"].to_numpy(float),
        out["final_prediction"].to_numpy(float),
        epsilon=float(cfg["target"].get("epsilon", 1e-12)),
        clip_min=float(cfg["evaluation"].get("clip_logvol_min", -20.0)),
        clip_max=float(cfg["evaluation"].get("clip_logvol_max", 20.0)),
    )
    out["qlike_loss"] = qlike
    out["clipped_predictions"] = int(clipped)
    out["residual_actual"] = out["actual_logvol"].astype(float) - out["p_prediction"].astype(float)
    out["residual_prediction"] = out["final_residual_prediction"].astype(float)
    out["spike_flag"] = out["actual_logvol"].astype(float) >= out["actual_logvol"].astype(float).quantile(float(cfg["evaluation"].get("spike_quantile", 0.90)))
    return out


def prediction_level_from_event_rows(event_rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    group_cols = ["date", "target_date", "target_ticker", "horizon", "split", "fold_id", "seed", "model"]
    base_cols = ["actual_logvol", "p_prediction", "stock_residual_prediction", "stock_prediction"]
    agg = event_rows.groupby(group_cols, as_index=False).agg(
        gated_correction=("gated_correction", "sum"),
        **{col: (col, "first") for col in base_cols},
    )
    agg = agg.rename(columns={"target_ticker": "ticker"})
    agg["final_residual_prediction"] = agg["stock_residual_prediction"].astype(float) + agg["gated_correction"].astype(float)
    agg["final_prediction"] = agg["p_prediction"].astype(float) + agg["final_residual_prediction"].astype(float)
    return add_qlike(agg, cfg)


def write_metrics(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics_by_model": summarize_predictions(predictions, ["split", "model"]),
        "metrics_by_ticker": summarize_predictions(predictions, ["split", "model", "ticker"]),
        "metrics_by_horizon": summarize_predictions(predictions, ["split", "model", "horizon"]),
        "metrics_by_fold_seed": summarize_predictions(predictions, ["split", "model", "fold_id", "seed"]),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def decide_step7(metrics_by_model: pd.DataFrame) -> tuple[str, list[str]]:
    val = metrics_by_model.loc[metrics_by_model["split"].astype(str).eq("validation")].set_index("model")
    reasons: list[str] = []
    if "S0_StockOnly_G5" not in val.index:
        return "UNAVAILABLE", ["S0_StockOnly_G5 is missing from validation metrics."]
    s0 = float(val.loc["S0_StockOnly_G5", "qlike"])
    candidates = val.drop(index=["S0_StockOnly_G5"], errors="ignore")
    if candidates.empty:
        return "UNAVAILABLE", ["No gated candidate model is available."]
    best_name = str(candidates["qlike"].astype(float).idxmin())
    best = float(candidates.loc[best_name, "qlike"])
    reasons.append(f"Best gated model {best_name} validation QLIKE={best:.6f}; S0={s0:.6f}.")
    if best < s0:
        return "GO", reasons
    if best < s0 * 1.02:
        return "PROTECTIVE GATE ONLY", reasons
    return "NO-GO", reasons


def write_report(output_dir: str | Path, report_path: str | Path, decision: str, reasons: list[str]) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 7 Stock-Specific News Impact Pilot Report",
        "",
        "This pilot report is generated from validation artifacts only. It is not a full-grid or locked-test result.",
        "",
        f"Decision: {decision}",
        "",
        "## Evidence",
        "",
        *[f"- {r}" for r in reasons],
        "",
        "## Scope",
        "",
        "- Stock backbone is treated as frozen.",
        "- Step 6 news correction is used as a correction proxy.",
        "- Pilot modes are intentionally small and should be expanded only if gates show signal.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(predictions: pd.DataFrame, output_root: str | Path) -> None:
    fig_dir = Path(output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return
    predictions.groupby("model")["qlike_loss"].mean().sort_values().plot(kind="bar", title="Step 7 pilot QLIKE by model")
    plt.tight_layout()
    plt.savefig(fig_dir / "step7_qlike_by_model.png")
    plt.close()
