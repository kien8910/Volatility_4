from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cross_stock_diagnostics import dependency_summary, lagged_cross_correlations, raw_correlation_matrix, residual_correlation_matrix
from .evaluator import p_model_performance
from .p_model_loader import load_p_model_config, read_yaml
from .residual_builder import add_har_features, date_split_labels, load_panel
from .residual_diagnostics import aggregate_diagnostics, diagnostics_by_ticker_split
from .spectral_diagnostics import acf_values
from .walk_forward_predictor import build_state_residuals, walk_forward_predictions


def _write_figures(state: pd.DataFrame, by_ticker: pd.DataFrame, corr: pd.DataFrame, config: dict, root: Path) -> None:
    figdir = root / config["output"]["figures_dir"]
    figdir.mkdir(parents=True, exist_ok=True)
    tickers = config["data"]["tickers"]
    sample = state.loc[state["ticker"].isin(tickers)].sort_values("date")
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, 18), sharex=True)
    for ax, ticker in zip(axes, tickers):
        part = sample.loc[sample["ticker"].eq(ticker)]
        ax.plot(part["date"], part["actual_logvol_gk"], label="raw", linewidth=0.8)
        ax.plot(part["date"], part["residual_state_h1"], label="residual", linewidth=0.8)
        ax.set_title(ticker, fontsize=8)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "step3_raw_vs_residual_time_series.png", dpi=150)
    plt.close()

    lags = config["diagnostics"]["acf_max_lag"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw_acf = np.nanmean([acf_values(state.loc[state.ticker.eq(t), "actual_logvol_gk"], lags) for t in tickers], axis=0)
    res_acf = np.nanmean([acf_values(state.loc[state.ticker.eq(t), "residual_state_h1"], lags) for t in tickers], axis=0)
    axes[0].bar(np.arange(1, lags + 1), raw_acf)
    axes[0].set_title("Raw ACF")
    axes[1].bar(np.arange(1, lags + 1), res_acf)
    axes[1].set_title("Residual ACF")
    plt.tight_layout()
    plt.savefig(figdir / "step3_raw_vs_residual_acf.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for label, col in [("raw", "actual_logvol_gk"), ("residual", "residual_state_h1")]:
        arr = state[col].dropna().to_numpy(dtype=float)
        power = np.abs(np.fft.rfft(arr - arr.mean())) ** 2
        plt.plot(power[1:200], label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "step3_raw_vs_residual_periodogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for ticker in tickers:
        state.loc[state.ticker.eq(ticker), "residual_state_h1"].plot(kind="kde", alpha=0.5, label=ticker)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figdir / "step3_residual_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    by_ticker.pivot(index="ticker", columns="base_split", values="variance_ratio").plot(kind="bar", ax=plt.gca())
    plt.tight_layout()
    plt.savefig(figdir / "step3_residual_variance_ratio.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.colorbar(label="corr")
    plt.tight_layout()
    plt.savefig(figdir / "step3_residual_correlation_heatmap.png", dpi=150)
    plt.close()


def _decision(overall: pd.DataFrame, dep: dict, failures: pd.DataFrame, config: dict) -> str:
    min_reduction = float(config["diagnostics"]["minimum_reduction"])
    acf_ok = float(overall["median_acf_energy_reduction"].iloc[0]) >= min_reduction
    lfr_ok = float(overall["median_lfr_reduction"].iloc[0]) >= min_reduction
    dep_ok = dep.get("significant_lagged_pairs_p05", 0) > 0 or dep.get("mean_abs_offdiag_residual_corr", 0) > 0.05
    if len(failures):
        return "NO-GO"
    if (acf_ok or lfr_ok) and dep_ok:
        return "GO"
    return "CONDITIONAL GO"


def _write_report(path: Path, p_config: dict, overall: pd.DataFrame, by_split: pd.DataFrame, dep: dict, perf: pd.DataFrame, failures: pd.DataFrame, decision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    garch = "Yes" if p_config.get("fallback_used", False) else "No"
    best_perf = perf.groupby("horizon")["qlike"].mean().reset_index()
    lines = [
        "# Step 3 Residual Report",
        "",
        f"- P model: {p_config.get('model_name', 'HAR-Ridge')}",
        f"- Fallback to HAR-Ridge: {garch}",
        "- Residual construction: pseudo-out-of-sample expanding walk-forward.",
        f"- Failure rows: {len(failures)}",
        "",
        "## Overall Diagnostics",
        "",
        overall.to_markdown(index=False),
        "",
        "## Diagnostics By Split",
        "",
        by_split.to_markdown(index=False),
        "",
        "## P Model QLIKE By Horizon",
        "",
        best_perf.to_markdown(index=False),
        "",
        "## Cross-Stock Dependency",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in dep.items()]
    lines += ["", "## Decision", "", decision]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str, root: Path) -> dict:
    config = read_yaml(config_path)
    step3_dir = root / config["output"]["step3_dir"]
    tables_dir = root / config["output"]["tables_dir"]
    for p in [step3_dir, tables_dir, root / "data/processed"]:
        p.mkdir(parents=True, exist_ok=True)
    panel, split_manifest, folds = load_panel(config, root)
    horizons = [int(h) for h in config["target"]["horizons"]]
    panel = add_har_features(panel, horizons)
    p_config = load_p_model_config(config, root)
    preds = walk_forward_predictions(
        panel,
        split_manifest,
        folds,
        p_config,
        horizons,
        int(config["p_model"]["initial_training_days"]),
        int(config["p_model"]["refit_frequency"]),
    )
    labels = date_split_labels(split_manifest, folds)
    state = build_state_residuals(preds, panel, labels)
    targets = preds[["date", "target_date", "ticker", "horizon", "fold_id", "base_split", "actual_target", "p_prediction", "residual_target", "model_name", "is_oos", "max_training_target_date", "training_observations"]].copy()
    failures = pd.DataFrame(columns=["step", "ticker", "horizon", "date", "message"])
    if not state["is_oos"].eq(1).all() or not targets["is_oos"].eq(1).all():
        failures.loc[len(failures)] = ["oos", "", "", "", "Non-OOS rows found."]
    if not targets["target_date"].gt(targets["date"]).all():
        failures.loc[len(failures)] = ["alignment", "", "", "", "target_date <= forecast origin."]
    if (targets["max_training_target_date"] > targets["date"]).any():
        failures.loc[len(failures)] = ["leakage", "", "", "", "Training target date after origin."]

    preds.to_parquet(step3_dir / "oos_p_predictions.parquet", index=False)
    state.to_parquet(root / "data/processed/step3_residual_state.parquet", index=False)
    targets.to_parquet(root / "data/processed/step3_residual_targets.parquet", index=False)

    eps = float(config["target"]["epsilon"])
    by_ticker = diagnostics_by_ticker_split(state, int(config["diagnostics"]["acf_energy_lag"]), int(config["diagnostics"]["acf_max_lag"]), float(config["diagnostics"]["low_frequency_fraction"]), eps)
    overall, by_split = aggregate_diagnostics(by_ticker)
    perf = p_model_performance(targets, eps)
    corr = residual_correlation_matrix(state)
    raw_corr = raw_correlation_matrix(state)
    lagged = lagged_cross_correlations(state, [int(x) for x in config["diagnostics"]["cross_correlation_lags"]])
    dep = dependency_summary(corr, raw_corr, lagged)
    decision = _decision(overall, dep, failures, config)

    by_ticker.to_csv(tables_dir / "step3_residual_diagnostics_by_ticker.csv", index=False)
    by_split.to_csv(tables_dir / "step3_residual_diagnostics_by_split.csv", index=False)
    overall.to_csv(tables_dir / "step3_residual_diagnostics.csv", index=False)
    perf.to_csv(tables_dir / "step3_p_model_performance.csv", index=False)
    failures.to_csv(tables_dir / "step3_failures.csv", index=False)
    corr.to_csv(step3_dir / "residual_correlation_matrix.csv")
    lagged.to_csv(step3_dir / "residual_lagged_cross_correlation.csv", index=False)
    (step3_dir / "dependency_summary.json").write_text(json.dumps(dep, indent=2, default=str), encoding="utf-8")
    _write_figures(state, by_ticker, corr, config, root)
    _write_report(root / config["output"]["report_path"], p_config, overall, by_split, dep, perf, failures, decision)
    summary = {"decision": decision, "state_rows": int(len(state)), "target_rows": int(len(targets)), "failures": int(len(failures)), **dep}
    (step3_dir / "step3_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step3_residual.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(run(args.config, Path(args.root).resolve()), indent=2, default=str))


if __name__ == "__main__":
    main()
