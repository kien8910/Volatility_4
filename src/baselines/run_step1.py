from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import add_time_series_features, dev_test_dates, load_config, load_step1_inputs, split_dates_for_fold
from .metrics import dm_test, holm_adjust, qlike, summarize_predictions
from .models import (
    FitResult,
    ar_model,
    dlinear_history,
    garch_family,
    har_ols,
    har_ridge,
    hyper_json,
    last_value,
    linear_history,
    mean_global,
    mean_ticker,
    mlp_history,
)


def model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"model": "B0_GlobalMean", "scope": "global", "fn": mean_global, "params": {}, "seed": 0},
        {"model": "B0_TickerMean", "scope": "global", "fn": mean_ticker, "params": {}, "seed": 0},
        {"model": "B1_LastValue", "scope": "global", "fn": last_value, "params": {}, "seed": 0},
        {"model": "B3_HAR_OLS", "scope": "ticker", "fn": har_ols, "params": {}, "seed": 0},
    ]
    for p in config["models"]["ar_lags"]:
        specs.append({"model": "B2_AR", "scope": "ticker", "fn": ar_model, "params": {"p": int(p)}, "seed": 0})
    for alpha in config["models"]["ridge_alpha"]:
        specs.append({"model": "B4_HAR_Ridge", "scope": "ticker", "fn": har_ridge, "params": {"alpha": float(alpha)}, "seed": 0})
    for lb in config["features"]["lookbacks"]:
        specs.append({"model": "B6_Linear", "scope": "ticker", "fn": linear_history, "params": {"lookback": int(lb)}, "seed": 0})
    mlp_cfg = config["models"]["mlp"]
    if mlp_cfg.get("enabled", True):
        for lb in mlp_cfg["lookbacks"]:
            for seed in mlp_cfg["seeds"]:
                specs.append({
                    "model": "B6_MLP",
                    "scope": "ticker",
                    "fn": mlp_history,
                    "params": {"lookback": int(lb), "hidden": tuple(mlp_cfg["hidden_layer_sizes"]), "max_iter": int(mlp_cfg["max_iter"]), "seed": int(seed)},
                    "seed": int(seed),
                })
    dl_cfg = config["models"]["dlinear"]
    if dl_cfg.get("enabled", True):
        for lb in dl_cfg["lookbacks"]:
            for seed in dl_cfg["seeds"]:
                specs.append({
                    "model": "B7_DLinear",
                    "scope": "ticker",
                    "fn": dlinear_history,
                    "params": {
                        "lookback": int(lb),
                        "seed": int(seed),
                        "max_epochs": int(dl_cfg["max_epochs"]),
                        "patience": int(dl_cfg["patience"]),
                        "learning_rate": float(dl_cfg["learning_rate"]),
                    },
                    "seed": int(seed),
                })
    garch_cfg = config["models"]["garch"]
    if garch_cfg.get("enabled", True):
        for family in garch_cfg["families"]:
            for distribution in garch_cfg["distributions"]:
                specs.append({"model": "B5_GARCH_Family", "scope": "ticker", "fn": garch_family, "params": {"family": family, "distribution": distribution}, "seed": 0})
    return specs


def _base_rows(part: pd.DataFrame, horizon: int) -> pd.DataFrame:
    cols = ["date", "ticker", f"target_date_h{horizon}", f"target_logvol_gk_h{horizon}", "logvol_gk", "log_return"]
    return part.loc[part[f"valid_h{horizon}"], cols + ["har_d", "har_w", "har_m"] + [c for c in part.columns if c.startswith("lag_")]].copy()


def _prediction_frame(val: pd.DataFrame, result: FitResult, split: str, model: str, horizon: int, fold_id: int, seed: int, hyper: dict, spike_threshold: float, epsilon: float, exp_clip: tuple[float, float]) -> pd.DataFrame:
    out = pd.DataFrame({
        "split": split,
        "fold_id": fold_id,
        "model": model,
        "ticker": val["ticker"].to_numpy(),
        "horizon": horizon,
        "seed": seed,
        "hyperparams": hyper_json(hyper),
        "feature_date": val["date"].to_numpy(),
        "target_date": val[f"target_date_h{horizon}"].to_numpy(),
        "y_true": val[f"target_logvol_gk_h{horizon}"].to_numpy(dtype=float),
        "y_pred": result.y_pred.astype(float),
        "status": result.status,
        "training_time_sec": result.training_time_sec,
        "inference_time_sec": result.inference_time_sec,
        "param_count": result.param_count,
    })
    losses, clipped = qlike(out["y_true"], out["y_pred"], epsilon, exp_clip)
    out["qlike_loss"] = losses
    out["pred_exp_clipped"] = clipped.astype(int)
    out["is_spike"] = out["y_true"] > spike_threshold
    return out


def _failure(model: str, split: str, horizon: int, fold_id: int, ticker: str, seed: int, hyper: dict, result: FitResult) -> dict[str, Any]:
    return {
        "split": split,
        "fold_id": fold_id,
        "ticker": ticker,
        "horizon": horizon,
        "model": model,
        "seed": seed,
        "hyperparams": hyper_json(hyper),
        "status": result.status,
        "message": result.message,
    }


def run_spec(spec: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, target: str, horizon: int, epsilon: float) -> FitResult:
    params = dict(spec["params"])
    params.update({"horizon": horizon, "epsilon": epsilon})
    try:
        return spec["fn"](train=train, val=val, target=target, **params)
    except Exception as exc:  # noqa: BLE001 - every model failure is logged row-wise.
        return FitResult(np.full(len(val), np.nan), params, status="failed", message=repr(exc))


def validation_predictions(df: pd.DataFrame, folds: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = float(config["target"]["epsilon"])
    exp_clip = tuple(config["evaluation"]["exp_clip"])
    predictions = []
    failures = []
    specs = model_specs(config)
    for fold_id in sorted(folds["fold_id"].unique()):
        train_dates, val_dates = split_dates_for_fold(folds, int(fold_id))
        for horizon in config["target"]["horizons"]:
            target = f"target_logvol_gk_h{horizon}"
            train_all = _base_rows(df.loc[df["date"].isin(train_dates)], horizon)
            val_all = _base_rows(df.loc[df["date"].isin(val_dates)], horizon)
            spike_threshold = float(train_all[target].quantile(float(config["evaluation"]["spike_quantile"])))
            for spec in specs:
                if spec["scope"] == "global":
                    result = run_spec(spec, train_all, val_all, target, horizon, eps)
                    if result.status == "ok":
                        predictions.append(_prediction_frame(val_all, result, "validation", spec["model"], horizon, int(fold_id), spec["seed"], result.hyperparams, spike_threshold, eps, exp_clip))
                    else:
                        failures.append(_failure(spec["model"], "validation", horizon, int(fold_id), "ALL", spec["seed"], spec["params"], result))
                    continue
                for ticker in config["data"]["tickers"]:
                    tr = train_all.loc[train_all["ticker"].eq(ticker)]
                    va = val_all.loc[val_all["ticker"].eq(ticker)]
                    if len(tr) == 0 or len(va) == 0:
                        failures.append({"split": "validation", "fold_id": fold_id, "ticker": ticker, "horizon": horizon, "model": spec["model"], "seed": spec["seed"], "hyperparams": hyper_json(spec["params"]), "status": "failed", "message": "Empty train or validation slice."})
                        continue
                    result = run_spec(spec, tr, va, target, horizon, eps)
                    if result.status == "ok" and np.isfinite(result.y_pred).all():
                        predictions.append(_prediction_frame(va, result, "validation", spec["model"], horizon, int(fold_id), spec["seed"], result.hyperparams, spike_threshold, eps, exp_clip))
                    else:
                        failures.append(_failure(spec["model"], "validation", horizon, int(fold_id), ticker, spec["seed"], spec["params"], result))
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    fail = pd.DataFrame(failures)
    return pred, fail


def select_configs(validation_pred: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for (model, ticker, horizon), part in validation_pred.groupby(["model", "ticker", "horizon"]):
        summary = part.groupby(["hyperparams", "seed"], dropna=False)["qlike_loss"].mean().reset_index()
        best = summary.sort_values(["qlike_loss", "hyperparams", "seed"]).iloc[0]
        rows.append({"model": model, "ticker": ticker, "horizon": horizon, "hyperparams": best.hyperparams, "seed": int(best.seed), "validation_qlike": float(best.qlike_loss)})
    selected = pd.DataFrame(rows)
    for model in ["B0_GlobalMean", "B0_TickerMean", "B1_LastValue"]:
        existing = selected.loc[selected["model"].eq(model)]
        if existing.empty:
            continue
        for horizon in existing["horizon"].unique():
            template = existing.loc[existing["horizon"].eq(horizon)].iloc[0]
            for ticker in tickers:
                if not ((selected.model == model) & (selected.ticker == ticker) & (selected.horizon == horizon)).any():
                    selected = pd.concat([selected, pd.DataFrame([{**template.to_dict(), "ticker": ticker}])], ignore_index=True)
    return selected


def _spec_from_selection(specs: list[dict[str, Any]], model: str, hyperparams: str, seed: int) -> dict[str, Any] | None:
    for spec in specs:
        expected = dict(spec["params"])
        if spec["model"] in {"B0_GlobalMean", "B0_TickerMean", "B1_LastValue", "B3_HAR_OLS"}:
            expected = {}
        if hyper_json(expected) == hyperparams and int(spec["seed"]) == int(seed) and spec["model"] == model:
            return spec
    return None


def test_predictions(df: pd.DataFrame, split_manifest: pd.DataFrame, config: dict[str, Any], selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = float(config["target"]["epsilon"])
    exp_clip = tuple(config["evaluation"]["exp_clip"])
    dev_dates, test_dates = dev_test_dates(split_manifest)
    specs = model_specs(config)
    predictions = []
    failures = []
    for horizon in config["target"]["horizons"]:
        target = f"target_logvol_gk_h{horizon}"
        dev_all = _base_rows(df.loc[df["date"].isin(dev_dates)], horizon)
        test_all = _base_rows(df.loc[df["date"].isin(test_dates)], horizon)
        spike_threshold = float(dev_all[target].quantile(float(config["evaluation"]["spike_quantile"])))
        for _, sel in selected.loc[selected["horizon"].eq(horizon)].iterrows():
            spec = _spec_from_selection(specs, sel.model, sel.hyperparams, int(sel.seed))
            if spec is None:
                failures.append({"split": "test", "fold_id": 0, "ticker": sel.ticker, "horizon": horizon, "model": sel.model, "seed": int(sel.seed), "hyperparams": sel.hyperparams, "status": "failed", "message": "Selected spec not found."})
                continue
            if spec["scope"] == "global":
                if sel.ticker != config["data"]["tickers"][0]:
                    continue
                result = run_spec(spec, dev_all, test_all, target, horizon, eps)
                if result.status == "ok":
                    predictions.append(_prediction_frame(test_all, result, "test", spec["model"], horizon, 0, spec["seed"], result.hyperparams, spike_threshold, eps, exp_clip))
                else:
                    failures.append(_failure(spec["model"], "test", horizon, 0, "ALL", spec["seed"], spec["params"], result))
                continue
            tr = dev_all.loc[dev_all["ticker"].eq(sel.ticker)]
            te = test_all.loc[test_all["ticker"].eq(sel.ticker)]
            result = run_spec(spec, tr, te, target, horizon, eps)
            if result.status == "ok" and np.isfinite(result.y_pred).all():
                predictions.append(_prediction_frame(te, result, "test", spec["model"], horizon, 0, int(sel.seed), result.hyperparams, spike_threshold, eps, exp_clip))
            else:
                failures.append(_failure(spec["model"], "test", horizon, 0, sel.ticker, int(sel.seed), json.loads(sel.hyperparams), result))
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    fail = pd.DataFrame(failures)
    return pred, fail


def ranking_table(test_pred: pd.DataFrame, epsilon: float, exp_clip: tuple[float, float]) -> pd.DataFrame:
    if test_pred.empty:
        return pd.DataFrame()
    rows = []
    baseline_keys = ["ticker", "horizon", "feature_date", "target_date"]
    for horizon, hpart in test_pred.groupby("horizon"):
        base_lv = hpart.loc[hpart["model"].eq("B1_LastValue"), baseline_keys + ["qlike_loss"]].rename(columns={"qlike_loss": "last_loss"})
        base_har = hpart.loc[hpart["model"].eq("B3_HAR_OLS"), baseline_keys + ["qlike_loss"]].rename(columns={"qlike_loss": "har_loss"})
        for model, part in hpart.groupby("model"):
            merged = part.merge(base_lv, on=baseline_keys, how="inner").merge(base_har, on=baseline_keys, how="inner")
            if merged.empty:
                continue
            dm_last, p_last = dm_test(merged["qlike_loss"].to_numpy(), merged["last_loss"].to_numpy(), block_lag=max(5, int(horizon)))
            dm_har, p_har = dm_test(merged["qlike_loss"].to_numpy(), merged["har_loss"].to_numpy(), block_lag=max(5, int(horizon)))
            rows.append({
                "horizon": horizon,
                "model": model,
                "qlike": float(part["qlike_loss"].mean()),
                "rank": np.nan,
                "loss_diff_vs_last_value": float((merged["qlike_loss"] - merged["last_loss"]).mean()),
                "win_rate_vs_last_value": float((merged["qlike_loss"] < merged["last_loss"]).mean()),
                "dm_stat_vs_last_value": dm_last,
                "dm_pvalue_vs_last_value": p_last,
                "loss_diff_vs_har_ols": float((merged["qlike_loss"] - merged["har_loss"]).mean()),
                "win_rate_vs_har_ols": float((merged["qlike_loss"] < merged["har_loss"]).mean()),
                "dm_stat_vs_har_ols": dm_har,
                "dm_pvalue_vs_har_ols": p_har,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out.groupby("horizon")["qlike"].rank(method="dense").astype(int)
    out["holm_pvalue_vs_last_value"] = out.groupby("horizon", group_keys=False)["dm_pvalue_vs_last_value"].apply(holm_adjust)
    out["holm_pvalue_vs_har_ols"] = out.groupby("horizon", group_keys=False)["dm_pvalue_vs_har_ols"].apply(holm_adjust)
    return out.sort_values(["horizon", "rank", "model"])


def write_figures(test_pred: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if test_pred.empty:
        return
    plt.figure(figsize=(10, 5))
    test_pred.groupby("model")["qlike_loss"].mean().sort_values().plot(kind="bar")
    plt.ylabel("Mean QLIKE")
    plt.tight_layout()
    plt.savefig(figures_dir / "step1_qlike_by_model.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    test_pred.groupby("ticker")["qlike_loss"].mean().sort_values().plot(kind="bar")
    plt.ylabel("Mean QLIKE")
    plt.tight_layout()
    plt.savefig(figures_dir / "step1_qlike_by_ticker.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    test_pred.groupby(["horizon", "model"])["qlike_loss"].mean().unstack().plot(marker="o")
    plt.ylabel("Mean QLIKE")
    plt.tight_layout()
    plt.savefig(figures_dir / "step1_qlike_by_horizon.png", dpi=150)
    plt.close()

    examples = test_pred.loc[test_pred["ticker"].isin(["NVDA", "AMD"]) & test_pred["horizon"].eq(1) & test_pred["model"].isin(["B1_LastValue", "B3_HAR_OLS"])]
    plt.figure(figsize=(11, 5))
    for (ticker, model), part in examples.groupby(["ticker", "model"]):
        part = part.sort_values("feature_date").head(80)
        plt.plot(part["feature_date"], part["y_pred"], label=f"{ticker} {model}", alpha=0.8)
    truth = examples.loc[examples["model"].eq("B1_LastValue")].sort_values("feature_date").head(80)
    if len(truth):
        plt.plot(truth["feature_date"], truth["y_true"], label="actual", color="black", linewidth=1.5)
    plt.legend(fontsize=8)
    plt.ylabel("logvol_gk")
    plt.tight_layout()
    plt.savefig(figures_dir / "step1_predictions_examples.png", dpi=150)
    plt.close()

    base = test_pred.loc[test_pred["model"].eq("B1_LastValue"), ["ticker", "horizon", "feature_date", "target_date", "qlike_loss"]].rename(columns={"qlike_loss": "last_loss"})
    best_model = test_pred.groupby("model")["qlike_loss"].mean().sort_values().index[0]
    best = test_pred.loc[test_pred["model"].eq(best_model)].merge(base, on=["ticker", "horizon", "feature_date", "target_date"], how="inner")
    best = best.sort_values("feature_date")
    plt.figure(figsize=(10, 5))
    (best["qlike_loss"] - best["last_loss"]).cumsum().plot()
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Cumulative loss difference: {best_model} minus Last Value")
    plt.ylabel("Cumulative QLIKE difference")
    plt.tight_layout()
    plt.savefig(figures_dir / "step1_cumulative_loss_difference.png", dpi=150)
    plt.close()


def write_report(path: Path, config: dict[str, Any], audit: dict[str, Any], validation_summary: pd.DataFrame, test_summary: pd.DataFrame, ranking: pd.DataFrame, failures: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ticker_counts = audit["ticker_counts"]
    best = test_summary.sort_values("qlike").iloc[0] if len(test_summary) else None
    har = test_summary.loc[test_summary["model"].eq("B3_HAR_OLS"), "qlike"].mean()
    hist = test_summary.loc[test_summary["model"].isin(["B0_GlobalMean", "B0_TickerMean"]), "qlike"].mean()
    last = test_summary.loc[test_summary["model"].eq("B1_LastValue"), "qlike"].mean()
    decision = "GO" if len(test_summary) and failures.empty and np.isfinite(test_summary["qlike"]).all() and audit["max_abs_logvol_gk_error"] <= 1e-10 else "CONDITIONAL GO"
    if test_summary.empty:
        decision = "NO-GO"
    lines = [
        "# Step 1 Baseline Report",
        "",
        f"Group: {config['data']['group_name']}. Fixed tickers: {', '.join(config['data']['tickers'])}.",
        "",
        "## Data audit",
        "",
        f"- Target check rows: {audit['target_check_rows']}",
        f"- Max absolute GK log-volatility recomputation error: {audit['max_abs_logvol_gk_error']:.3e}",
        f"- GK variance raw <= 0 count: {audit['gk_nonpositive_count']}",
        "",
        "## Ticker observations",
        "",
        ticker_counts.to_markdown(index=False),
        "",
        "## Test result",
        "",
    ]
    if best is not None:
        lines += [
            f"- Best baseline by test QLIKE: {best['model']} at horizon {int(best['horizon'])}, QLIKE={best['qlike']:.6f}.",
            f"- Mean HAR-OLS QLIKE: {har:.6f}; Historical Mean QLIKE: {hist:.6f}; Last Value QLIKE: {last:.6f}.",
        ]
    if len(failures):
        lines.append(f"- Failure rows logged: {len(failures)}. See `results/tables/step1_failures.csv`.")
    else:
        lines.append("- Failure rows logged: 0.")
    lines += ["", "## Decision", "", decision]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str, root: Path) -> dict[str, Any]:
    start_all = time.perf_counter()
    config = load_config(config_path)
    tables_dir = root / config["output"]["tables_dir"]
    figures_dir = root / config["output"]["figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df, split_manifest, folds, audit = load_step1_inputs(config, root)
    df = add_time_series_features(df, config["features"]["lookbacks"], config["target"]["horizons"])
    validation_pred, validation_failures = validation_predictions(df, folds, config)
    selected = select_configs(validation_pred, config["data"]["tickers"])
    test_pred, test_failures = test_predictions(df, split_manifest, config, selected)
    failures = pd.concat([validation_failures, test_failures], ignore_index=True) if len(validation_failures) or len(test_failures) else pd.DataFrame(columns=["split", "fold_id", "ticker", "horizon", "model", "seed", "hyperparams", "status", "message"])
    eps = float(config["target"]["epsilon"])
    exp_clip = tuple(config["evaluation"]["exp_clip"])
    spike_q = float(config["evaluation"]["spike_quantile"])
    validation_summary = summarize_predictions(validation_pred, ["model", "horizon", "hyperparams", "seed"], spike_q, eps, exp_clip)
    test_summary = summarize_predictions(test_pred, ["model", "horizon"], spike_q, eps, exp_clip)
    metrics_by_ticker = summarize_predictions(test_pred, ["model", "ticker", "horizon"], spike_q, eps, exp_clip)
    metrics_by_horizon = summarize_predictions(test_pred, ["model", "horizon"], spike_q, eps, exp_clip)
    ranking = ranking_table(test_pred, eps, exp_clip)

    validation_pred.to_parquet(tables_dir / "step1_predictions_validation.parquet", index=False)
    test_pred.to_parquet(tables_dir / "step1_predictions_test.parquet", index=False)
    metrics_by_ticker.to_csv(tables_dir / "step1_metrics_by_ticker.csv", index=False)
    metrics_by_horizon.to_csv(tables_dir / "step1_metrics_by_horizon.csv", index=False)
    validation_summary.to_csv(tables_dir / "step1_validation_summary.csv", index=False)
    test_summary.to_csv(tables_dir / "step1_test_summary.csv", index=False)
    ranking.to_csv(tables_dir / "step1_model_ranking.csv", index=False)
    failures.to_csv(tables_dir / "step1_failures.csv", index=False)
    audit["ticker_counts"].to_csv(tables_dir / "step1_ticker_observations.csv", index=False)
    pd.DataFrame([{
        "target_check_rows": audit["target_check_rows"],
        "max_abs_logvol_gk_error": audit["max_abs_logvol_gk_error"],
        "gk_nonpositive_count": audit["gk_nonpositive_count"],
    }]).to_csv(tables_dir / "step1_target_audit.csv", index=False)
    write_figures(test_pred, figures_dir)
    write_report(root / config["output"]["report_path"], config, audit, validation_summary, test_summary, ranking, failures)
    summary = {
        "validation_prediction_rows": int(len(validation_pred)),
        "test_prediction_rows": int(len(test_pred)),
        "failure_rows": int(len(failures)),
        "elapsed_sec": time.perf_counter() - start_all,
        "best_test": test_summary.sort_values("qlike").head(1).to_dict("records"),
    }
    (tables_dir / "step1_run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step1_baselines.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(run(args.config, Path(args.root).resolve()), indent=2, default=str))


if __name__ == "__main__":
    main()
