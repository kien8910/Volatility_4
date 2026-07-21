from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph.metrics import mae, mse, summarize_predictions


def flatten_step6_predictions(
    raw: dict[str, np.ndarray],
    samples,
    coverage: pd.DataFrame,
    split_name: str,
    fold_id: int,
    seed: int,
    run_config: dict,
    spike_threshold: float,
) -> pd.DataFrame:
    rows = []
    sample_indices = raw["sample_index"]
    coverage_idx = coverage.copy()
    coverage_idx["date"] = pd.to_datetime(coverage_idx["date"])
    cov = coverage_idx.set_index(["date", "ticker"])
    for local_row, sample_idx in enumerate(sample_indices):
        origin = pd.Timestamp(samples.sample_dates[int(sample_idx)])
        for i, ticker in enumerate(samples.tickers):
            coverage_row = cov.loc[(origin, ticker)].to_dict() if (origin, ticker) in cov.index else {}
            for h_idx, horizon in enumerate(samples.horizons):
                actual = float(raw["actual_logvol"][local_row, i, h_idx])
                rows.append(
                    {
                        "date": origin,
                        "target_date": pd.Timestamp(samples.target_dates[int(sample_idx), h_idx]),
                        "ticker": ticker,
                        "horizon": int(horizon),
                        "split": split_name,
                        "fold_id": int(fold_id),
                        "seed": int(seed),
                        "model": run_config["model"],
                        "config_id": run_config["config_id"],
                        "ablation": run_config.get("ablation", ""),
                        "pooling_method": run_config.get("pooling_method", ""),
                        "actual_logvol": actual,
                        "p_prediction": float(raw["p_prediction"][local_row, i, h_idx]),
                        "residual_actual": float(raw["residual_actual"][local_row, i, h_idx]),
                        "stock_residual_prediction": float(raw["stock_residual_prediction"][local_row, i, h_idx]),
                        "news_residual_correction": float(raw["news_residual_correction"][local_row, i, h_idx]),
                        "final_residual_prediction": float(raw["final_residual_prediction"][local_row, i, h_idx]),
                        "residual_prediction": float(raw["final_residual_prediction"][local_row, i, h_idx]),
                        "final_prediction": float(raw["final_prediction"][local_row, i, h_idx]),
                        "qlike_loss": float(raw["qlike_loss"][local_row, i, h_idx]),
                        "has_macro": int(coverage_row.get("has_macro", 0)),
                        "has_sector": int(coverage_row.get("has_sector", 0)),
                        "has_target_company": int(coverage_row.get("has_target_company", 0)),
                        "has_related_company": int(coverage_row.get("has_related_company", 0)),
                        "has_filing": int(coverage_row.get("has_filing", 0)),
                        "has_any_dynamic_news": int(coverage_row.get("has_any_dynamic_news", 0)),
                        "spike_flag": bool(actual >= spike_threshold),
                    }
                )
    return pd.DataFrame(rows)


def _group_metric(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return summarize_predictions(predictions, group_cols)


def metrics_by_news_group(predictions: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_days": np.ones(len(predictions), dtype=bool),
        "any_news_days": predictions["has_any_dynamic_news"].astype(bool).to_numpy(),
        "no_news_days": ~predictions["has_any_dynamic_news"].astype(bool).to_numpy(),
        "macro_news_days": predictions["has_macro"].astype(bool).to_numpy(),
        "sector_news_days": predictions["has_sector"].astype(bool).to_numpy(),
        "target_company_news_days": predictions["has_target_company"].astype(bool).to_numpy(),
        "related_company_news_days": predictions["has_related_company"].astype(bool).to_numpy(),
        "filing_present_days": predictions["has_filing"].astype(bool).to_numpy(),
        "spike_days": predictions["spike_flag"].astype(bool).to_numpy(),
        "normal_days": ~predictions["spike_flag"].astype(bool).to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        sub = predictions.loc[mask]
        if sub.empty:
            continue
        table = summarize_predictions(sub, ["split", "model", "config_id"])
        table.insert(0, "news_group", name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_step6_metric_tables(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics_by_model": _group_metric(predictions, ["split", "model", "config_id"]),
        "metrics_by_ticker": _group_metric(predictions, ["split", "model", "config_id", "ticker"]),
        "metrics_by_horizon": _group_metric(predictions, ["split", "model", "config_id", "horizon"]),
        "metrics_by_fold_seed": _group_metric(predictions, ["split", "model", "config_id", "fold_id", "seed"]),
        "metrics_by_news_group": metrics_by_news_group(predictions),
        "hierarchy_ablation": _group_metric(predictions, ["split", "model", "ablation", "pooling_method"]),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def news_coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["has_macro", "has_sector", "has_target_company", "has_related_company", "has_filing", "has_any_dynamic_news"]:
        rows.append({"coverage": col, "rows": int(len(coverage)), "positive_rows": int(coverage[col].sum()), "rate": float(coverage[col].mean())})
    return pd.DataFrame(rows)


def decide_step6(metrics_by_model: pd.DataFrame) -> tuple[str, list[str]]:
    val = metrics_by_model.loc[metrics_by_model["split"] == "validation"].copy()
    if val.empty:
        return "UNAVAILABLE", ["No validation metrics were available."]
    baseline = val.loc[val["model"] == "stock_only"]
    news = val.loc[val["model"] != "stock_only"]
    if baseline.empty or news.empty:
        return "UNAVAILABLE", ["Both stock_only and at least one news model are required for a Step 6 decision."]
    n0 = float(baseline.sort_values("qlike").iloc[0]["qlike"])
    best_news = news.sort_values("qlike").iloc[0]
    delta = n0 - float(best_news["qlike"])
    reasons = [f"Best news config={best_news['config_id']} validation QLIKE={best_news['qlike']:.6f}; N0={n0:.6f}; improvement={delta:.6f}."]
    if delta > 0:
        return "NEWS-HELPFUL", reasons
    news_group = None
    return "NEWS-NOISY", reasons if news_group is None else reasons + [news_group]


def write_step6_report(output_dir: str | Path, report_path: str | Path, decision: str, reasons: list[str]) -> None:
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 6 Naive Hierarchical News Fusion Report",
        "",
        "This report is generated from Step 6 artifacts after you run the experiment. It does not invent results.",
        "",
        f"Decision: {decision}",
        "",
        "## Evidence",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Required outputs",
        "",
        f"- `{output_dir / 'predictions_validation.parquet'}`",
        f"- `{output_dir / 'predictions_test.parquet'}`",
        f"- `{output_dir / 'metrics_by_model.csv'}`",
        f"- `{output_dir / 'metrics_by_news_group.csv'}`",
        f"- `{output_dir / 'hierarchy_ablation.csv'}`",
        f"- `{output_dir / 'news_corrections.parquet'}`",
        f"- `{output_dir / 'embedding_statistics.csv'}`",
        "",
        "## Interpretation checklist",
        "",
        "- Verify N0 reproduces Step 4 before trusting news comparisons.",
        "- Treat filing as company context, not as daily event news.",
        "- Prefer validation evidence; locked test must be evaluated once after selection.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_step6_figures(predictions: pd.DataFrame, output_root: str | Path) -> None:
    fig_dir = Path(output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return
    plots = [
        ("step6_qlike_by_model.png", predictions.groupby("model")["qlike_loss"].mean().sort_values(), "Step 6 QLIKE by model"),
        ("step6_qlike_by_horizon.png", predictions.groupby("horizon")["qlike_loss"].mean().sort_index(), "Step 6 QLIKE by horizon"),
        ("step6_qlike_by_ticker.png", predictions.groupby("ticker")["qlike_loss"].mean().sort_values(), "Step 6 QLIKE by ticker"),
    ]
    for filename, series, title in plots:
        series.plot(kind="bar", title=title, figsize=(10, 4))
        plt.tight_layout()
        plt.savefig(fig_dir / filename)
        plt.close()
    corr = predictions["news_residual_correction"].astype(float)
    corr.plot(kind="hist", bins=60, title="Step 6 news correction distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "step6_news_correction_distribution.png")
    plt.close()
    pivot = predictions.pivot_table(index="date", columns="model", values="qlike_loss", aggfunc="mean").sort_index()
    if "stock_only" in pivot.columns:
        for col in pivot.columns:
            if col == "stock_only":
                continue
            (pivot["stock_only"] - pivot[col]).cumsum().plot(label=col)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.legend()
        plt.title("Cumulative loss difference vs N0")
        plt.tight_layout()
        plt.savefig(fig_dir / "step6_cumulative_loss_difference.png")
        plt.close()

