from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph.metrics import summarize_predictions


def flatten_predictions(
    raw: dict[str, np.ndarray],
    samples,
    tickers: list[str],
    horizons: list[int],
    split_name: str,
    fold_id: int,
    seed: int,
    model_id: str,
    spike_quantile: float,
) -> pd.DataFrame:
    rows = []
    sample_indices = raw["sample_index"]
    threshold = np.quantile(raw["actual_logvol"], spike_quantile)
    for local_row, sample_idx in enumerate(sample_indices):
        origin = pd.Timestamp(samples.sample_dates[int(sample_idx)])
        for i, ticker in enumerate(tickers):
            for h_idx, horizon in enumerate(horizons):
                rows.append(
                    {
                        "date": origin,
                        "target_date": pd.Timestamp(samples.target_dates[int(sample_idx), h_idx]),
                        "ticker": ticker,
                        "horizon": int(horizon),
                        "split": split_name,
                        "fold_id": int(fold_id),
                        "seed": int(seed),
                        "model": model_id,
                        "actual_logvol": float(raw["actual_logvol"][local_row, i, h_idx]),
                        "p_prediction": float(raw["p_prediction"][local_row, i, h_idx]),
                        "residual_actual": float(raw["residual_actual"][local_row, i, h_idx]),
                        "residual_prediction": float(raw["residual_prediction"][local_row, i, h_idx]),
                        "final_prediction": float(raw["final_prediction"][local_row, i, h_idx]),
                        "qlike_loss": float(raw["qlike_loss"][local_row, i, h_idx]),
                        "spike_flag": bool(raw["actual_logvol"][local_row, i, h_idx] >= threshold),
                    }
                )
    return pd.DataFrame(rows)


def write_metric_tables(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
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


def select_best_config(metrics_by_fold_seed: pd.DataFrame, prediction_rows: pd.DataFrame) -> pd.DataFrame:
    val = metrics_by_fold_seed.loc[metrics_by_fold_seed["split"] == "validation"].copy()
    if val.empty:
        raise ValueError("No validation metrics available for model selection.")
    summary = val.groupby("model", as_index=False)["qlike"].mean().sort_values("qlike")
    best_model = str(summary.iloc[0]["model"])
    return prediction_rows.loc[(prediction_rows["split"] == "validation") & (prediction_rows["model"] == best_model)].copy()


def decide_go_no_go(metrics_by_model: pd.DataFrame, reconstruction: pd.DataFrame | None) -> tuple[str, list[str]]:
    val = metrics_by_model.loc[metrics_by_model["split"] == "validation"].set_index("model")
    reasons: list[str] = []
    positive = 0
    def better(lhs: str, rhs: str, threshold: float = 0.0) -> bool:
        if lhs not in val.index or rhs not in val.index:
            return False
        lhs_q = float(val.loc[lhs, "qlike"])
        rhs_q = float(val.loc[rhs, "qlike"])
        return lhs_q <= rhs_q * (1.0 - threshold)

    if better("G5", "G1", 0.01):
        positive += 1
        reasons.append("G5 improves validation QLIKE over G1 by at least 1%.")
    if better("G5", "G2", 0.0) or better("G5", "G3", 0.0):
        positive += 1
        reasons.append("G5 beats at least one graph control, G2 or G3.")
    if better("G5", "G4", 0.0):
        positive += 1
        reasons.append("G5 beats raw learned graph G4.")
    if reconstruction is not None and not reconstruction.empty:
        rec = reconstruction.groupby("model", as_index=True)["mse"].mean()
        if {"G5", "G1", "G3"}.issubset(rec.index) and rec["G5"] < min(rec["G1"], rec["G3"]) * 0.95:
            positive += 1
            reasons.append("Masked reconstruction MSE improves over identity and random by at least 5%.")

    if positive >= 2:
        return "GO", reasons
    if positive == 1:
        return "CONDITIONAL GO", reasons
    return "NO-GO", reasons or ["G5 did not satisfy enough validation gates."]


def write_report(output_dir: str | Path, report_path: str | Path, decision: str, reasons: list[str], failures: pd.DataFrame) -> None:
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 4 Static Residual Graph Report",
        "",
        "This report is generated only when the Step 4 pipeline is executed on your data.",
        "",
        f"Decision: {decision}",
        "",
        "## Decision Evidence",
        "",
    ]
    lines.extend([f"- {reason}" for reason in reasons])
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- {output_dir / 'metrics_by_model.csv'}",
            f"- {output_dir / 'metrics_by_ticker.csv'}",
            f"- {output_dir / 'metrics_by_horizon.csv'}",
            f"- {output_dir / 'metrics_by_fold_seed.csv'}",
            f"- {output_dir / 'masked_reconstruction.csv'}",
            f"- {output_dir / 'graph_edges.csv'}",
            f"- {output_dir / 'graph_stability.csv'}",
            f"- {output_dir / 'failures.csv'}",
            "",
            "## Failures",
            "",
            f"Failure rows: {len(failures)}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(predictions: pd.DataFrame, output_root: str | Path) -> None:
    fig_dir = Path(output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    by_model = predictions.groupby("model")["qlike_loss"].mean().sort_values()
    by_model.plot(kind="bar", title="Step 4 QLIKE by model")
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_model_qlike.png")
    plt.close()

    by_ticker = predictions.groupby(["ticker", "model"])["qlike_loss"].mean().unstack("model")
    by_ticker.plot(kind="bar", title="Step 4 QLIKE by ticker", figsize=(12, 5))
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_qlike_by_ticker.png")
    plt.close()

    by_horizon = predictions.groupby(["horizon", "model"])["qlike_loss"].mean().unstack("model")
    by_horizon.plot(kind="bar", title="Step 4 QLIKE by horizon")
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_qlike_by_horizon.png")
    plt.close()

    pivot = predictions.pivot_table(index="date", columns="model", values="qlike_loss", aggfunc="mean").sort_index()
    if "G1" in pivot and "G5" in pivot:
        (pivot["G5"] - pivot["G1"]).cumsum().plot(title="Cumulative loss difference: G5 - G1")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(fig_dir / "step4_cumulative_loss_difference.png")
        plt.close()

