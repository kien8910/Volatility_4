from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph.metrics import summarize_predictions


STATE_LABELS = {0: "low_volatility", 1: "medium_volatility", 2: "high_volatility"}


def flatten_step5_predictions(raw, samples, tickers, horizons, split_name, fold_id, seed, run_config, spike_quantile):
    rows = []
    threshold = np.quantile(raw["actual_logvol"], spike_quantile)
    weights = raw["regime_weights"]
    for local_row, sample_idx in enumerate(raw["sample_index"]):
        sample_idx = int(sample_idx)
        origin = pd.Timestamp(samples.sample_dates[sample_idx])
        regime_weight_row = weights[local_row]
        regime_argmax = int(np.argmax(regime_weight_row)) + 1 if regime_weight_row.size else None
        for i, ticker in enumerate(tickers):
            for h_idx, horizon in enumerate(horizons):
                row = {
                    "date": origin,
                    "target_date": pd.Timestamp(samples.target_dates[sample_idx, h_idx]),
                    "ticker": ticker,
                    "horizon": int(horizon),
                    "split": split_name,
                    "fold_id": int(fold_id),
                    "seed": int(seed),
                    "model": run_config["model"],
                    "config_id": run_config["config_id"],
                    "K": int(run_config["K"]),
                    "ema_beta": float(run_config["ema_beta"]),
                    "gate_temperature": float(run_config.get("gate_temperature", np.nan)),
                    "actual_logvol": float(raw["actual_logvol"][local_row, i, h_idx]),
                    "p_prediction": float(raw["p_prediction"][local_row, i, h_idx]),
                    "residual_actual": float(raw["residual_actual"][local_row, i, h_idx]),
                    "residual_prediction": float(raw["residual_prediction"][local_row, i, h_idx]),
                    "final_prediction": float(raw["final_prediction"][local_row, i, h_idx]),
                    "qlike_loss": float(raw["qlike_loss"][local_row, i, h_idx]),
                    "market_state": STATE_LABELS[int(raw["market_state_code"][local_row])],
                    "spike_flag": bool(raw["actual_logvol"][local_row, i, h_idx] >= threshold),
                    "regime_argmax": regime_argmax,
                }
                for k in range(4):
                    row[f"regime_weight_{k+1}"] = float(regime_weight_row[k]) if k < len(regime_weight_row) and int(run_config["K"]) > 1 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def write_step5_metric_tables(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics_by_model": summarize_predictions(predictions, ["split", "model"]),
        "metrics_by_ticker": summarize_predictions(predictions, ["split", "model", "ticker"]),
        "metrics_by_horizon": summarize_predictions(predictions, ["split", "model", "horizon"]),
        "metrics_by_fold_seed": summarize_predictions(predictions, ["split", "model", "fold_id", "seed"]),
        "metrics_by_regime": summarize_predictions(predictions.dropna(subset=["regime_argmax"]), ["split", "model", "regime_argmax"]) if "regime_argmax" in predictions else pd.DataFrame(),
    }
    market = summarize_predictions(predictions, ["split", "model", "market_state"])
    tables["metrics_by_market_state"] = market
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def decide_step5(metrics_by_model: pd.DataFrame) -> tuple[str, list[str]]:
    val = metrics_by_model.loc[metrics_by_model["split"] == "validation"].set_index("model")
    reasons = []
    positive = 0
    if "S5-B0" not in val.index:
        return "NO-GO", ["S5-B0 baseline missing."]
    base = float(val.loc["S5-B0", "qlike"])
    for candidate in ["S5-E", "S5-R", "S5-RE"]:
        if candidate in val.index:
            q = float(val.loc[candidate, "qlike"])
            if q < base:
                positive += 1
                reasons.append(f"{candidate} improves validation QLIKE over S5-B0.")
            if "spike_qlike" in val.columns and q <= base and float(val.loc[candidate, "spike_qlike"]) < float(val.loc["S5-B0", "spike_qlike"]):
                positive += 1
                reasons.append(f"{candidate} improves spike-day QLIKE without degrading overall QLIKE.")
    if positive >= 2:
        return "GO", reasons
    if positive == 1:
        return "CONDITIONAL GO", reasons
    return "NO-GO", reasons or ["No Step 5 candidate improves over static Step 4 G5 baseline."]


def write_step5_report(output_dir: str | Path, report_path: str | Path, decision: str, reasons: list[str]) -> None:
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 5 EMA and Regime Graph Report",
        "",
        f"Decision: {decision}",
        "",
        "## Evidence",
        "",
    ]
    lines.extend([f"- {reason}" for reason in reasons])
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- {output_dir / 'metrics_by_model.csv'}",
            f"- {output_dir / 'ema_stability.csv'}",
            f"- {output_dir / 'regime_usage.csv'}",
            f"- {output_dir / 'graph_diversity.csv'}",
            "",
            "Graph/regime assignments are predictive diagnostics, not causal claims.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_step5_figures(predictions: pd.DataFrame, output_root: str | Path) -> None:
    fig_dir = Path(output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    predictions.groupby("model")["qlike_loss"].mean().sort_values().plot(kind="bar", title="Step 5 QLIKE by model")
    plt.tight_layout()
    plt.savefig(fig_dir / "step5_qlike_by_model.png")
    plt.close()
    predictions.groupby(["horizon", "model"])["qlike_loss"].mean().unstack("model").plot(kind="bar", title="Step 5 QLIKE by horizon")
    plt.tight_layout()
    plt.savefig(fig_dir / "step5_qlike_by_horizon.png")
    plt.close()
    predictions.groupby(["market_state", "model"])["qlike_loss"].mean().unstack("model").plot(kind="bar", title="Step 5 QLIKE by market state")
    plt.tight_layout()
    plt.savefig(fig_dir / "step5_qlike_by_market_state.png")
    plt.close()
    pivot = predictions.pivot_table(index="date", columns="model", values="qlike_loss", aggfunc="mean").sort_index()
    if "S5-B0" in pivot:
        for col in pivot.columns:
            if col != "S5-B0":
                (pivot["S5-B0"] - pivot[col]).cumsum().plot(label=col)
        plt.legend()
        plt.title("Cumulative loss difference vs S5-B0")
        plt.tight_layout()
        plt.savefig(fig_dir / "step5_cumulative_loss_difference.png")
        plt.close()

