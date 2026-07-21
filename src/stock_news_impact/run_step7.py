from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.graph import SEMICONDUCTOR_TICKERS
from src.graph.data_loader import load_config
from src.graph.reproducibility import resolve_device, seed_everything
from src.stock_news_impact.abnormal_response import build_abnormal_response
from src.stock_news_impact.diagnostics import common_news_impact_diagnostics, gate_diagnostics
from src.stock_news_impact.event_builder import build_news_events, load_news_features_for_step7, validate_events
from src.stock_news_impact.event_features_builder import build_gate_feature_frame
from src.stock_news_impact.event_stock_pairs import build_event_stock_pairs, validate_pairs
from src.stock_news_impact.evaluator import decide_step7, prediction_level_from_event_rows, write_figures, write_metrics, write_report
from src.stock_news_impact.frozen_branches import choose_step6_news_branch
from src.stock_news_impact.oracle import oracle_diagnostic
from src.stock_news_impact.placebo_tests import wrong_ticker_placebo
from src.stock_news_impact.trainer import Step7RunConfig, train_gate_model
from src.stock_news_impact.treatment_control import simple_treatment_control_did


def setup_logger(log_dir: str | Path, mode: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step7")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(Path(log_dir) / f"{mode}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def atomic_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def atomic_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_step6_predictions(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data"]["step6_predictions_path"])
    if not path.exists():
        raise FileNotFoundError(f"Step 7 pilot requires Step 6 validation predictions: {path}")
    pred = pd.read_parquet(path)
    pred["date"] = pd.to_datetime(pred["date"])
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    return pred


def should_force_rebuild(cfg: dict) -> bool:
    return bool(cfg.get("runtime", {}).get("force_rebuild", False))


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["hierarchy", "event_scope", "events"])
    return (
        events.groupby(["hierarchy", "event_scope"], as_index=False)
        .agg(events=("event_id", "nunique"))
        .sort_values(["hierarchy", "event_scope"])
    )


def pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["hierarchy", "event_scope", "is_direct_target", "pairs", "events", "target_stocks"])
    return (
        pairs.groupby(["hierarchy", "event_scope", "is_direct_target"], as_index=False)
        .agg(pairs=("event_id", "size"), events=("event_id", "nunique"), target_stocks=("target_ticker", "nunique"))
        .sort_values(["hierarchy", "event_scope", "is_direct_target"])
    )


def log_summary(logger: logging.Logger, title: str, summary: pd.DataFrame) -> None:
    logger.info("%s:\n%s", title, summary.to_string(index=False) if not summary.empty else "<empty>")


def mode_validate_data(cfg: dict, logger: logging.Logger) -> None:
    if list(cfg["data"]["tickers"]) != SEMICONDUCTOR_TICKERS:
        raise ValueError(f"Step 7 ticker list must exactly match {SEMICONDUCTOR_TICKERS}")
    pred = load_step6_predictions(cfg)
    if pred.loc[pred["model"].astype(str).eq("stock_only")].empty:
        raise ValueError("Step 6 predictions must contain stock_only baseline rows.")
    features = load_news_features_for_step7(cfg)
    logger.info("Validated Step 7 pilot data step6_rows=%s news_feature_rows=%s", len(pred), len(features))


def mode_build_events(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    out_dir = Path(cfg["experiment"]["output_dir"])
    features = load_news_features_for_step7(cfg)
    events = build_news_events(features, cfg)
    validate_events(events)
    hierarchy_caps = cfg.get("pilot", {}).get("max_events_per_date_by_hierarchy") or {}
    if hierarchy_caps:
        capped = []
        for hierarchy, cap in hierarchy_caps.items():
            sub = events.loc[events["hierarchy"].astype(str).eq(str(hierarchy))]
            if cap is not None:
                sub = sub.groupby("date", group_keys=False).head(int(cap))
            capped.append(sub)
        configured = set(str(k) for k in hierarchy_caps)
        uncapped = events.loc[~events["hierarchy"].astype(str).isin(configured)]
        if not uncapped.empty:
            capped.append(uncapped)
        events = pd.concat(capped, ignore_index=True).sort_values(["date", "hierarchy", "event_id"]).reset_index(drop=True)
    else:
        max_events = int(cfg.get("pilot", {}).get("max_events_per_date", 0) or 0)
        if max_events > 0:
            events = events.groupby("date", group_keys=False).head(max_events).reset_index(drop=True)
    atomic_parquet(events, "data/processed/step7_news_events.parquet")
    atomic_parquet(events, out_dir / "step7_news_events.parquet")
    summary = event_summary(events)
    atomic_csv(summary, out_dir / "event_summary.csv")
    logger.info("Built Step 7 events rows=%s", len(events))
    log_summary(logger, "Step 7 event summary by hierarchy", summary)
    return events


def mode_build_pairs(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    out_dir = Path(cfg["experiment"]["output_dir"])
    events_path = Path("data/processed/step7_news_events.parquet")
    events = pd.read_parquet(events_path) if events_path.exists() and not should_force_rebuild(cfg) else mode_build_events(cfg, logger)
    pred = load_step6_predictions(cfg)
    stock = pred.loc[pred["model"].astype(str).eq("stock_only")].copy()
    pairs = build_event_stock_pairs(
        events,
        stock,
        list(cfg["data"]["tickers"]),
        spillover_mode=str(cfg["search"]["spillover_modes"][0]),
        max_pairs=cfg.get("pilot", {}).get("max_pairs"),
    )
    validate_pairs(pairs)
    atomic_parquet(pairs, "data/processed/step7_event_stock_pairs.parquet")
    atomic_parquet(pairs, out_dir / "step7_event_stock_pairs.parquet")
    summary = pair_summary(pairs)
    atomic_csv(summary, out_dir / "event_stock_pair_summary.csv")
    logger.info("Built Step 7 event-stock pairs rows=%s", len(pairs))
    log_summary(logger, "Step 7 event-stock pair summary by hierarchy", summary)
    return pairs


def mode_build_features(cfg: dict, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out_dir = Path(cfg["experiment"]["output_dir"])
    events_path = Path("data/processed/step7_news_events.parquet")
    pairs_path = Path("data/processed/step7_event_stock_pairs.parquet")
    events = pd.read_parquet(events_path) if events_path.exists() and not should_force_rebuild(cfg) else mode_build_events(cfg, logger)
    pairs = pd.read_parquet(pairs_path) if pairs_path.exists() and not should_force_rebuild(cfg) else mode_build_pairs(cfg, logger)
    pred = load_step6_predictions(cfg)
    selected = choose_step6_news_branch(cfg)
    abnormal = build_abnormal_response(pred)
    atomic_parquet(abnormal, out_dir / "abnormal_volatility_response.parquet")
    frame, columns = build_gate_feature_frame(events, pairs, pred, selected, cfg)
    atomic_parquet(frame, out_dir / "gate_features.parquet")
    train_utility = frame.loc[frame["utility_label"].astype(int).ne(-1)].copy()
    atomic_parquet(train_utility, out_dir / "utility_labels_train.parquet")
    atomic_csv(oracle_diagnostic(frame), out_dir / "oracle_diagnostics.csv")
    atomic_csv(simple_treatment_control_did(frame), out_dir / "did_diagnostics.csv")
    logger.info("Built Step 7 gate features rows=%s", len(frame))
    log_summary(logger, "Step 7 gate feature summary by hierarchy", pair_summary(frame))
    return frame, columns


def enumerate_run_configs(cfg: dict) -> list[Step7RunConfig]:
    models = list(cfg["search"].get("include_models") or ["S0_StockOnly_G5", "S2_FixedSmallGate", "S4_FactorizedGate"])
    runs: list[Step7RunConfig] = []
    for model in models:
        if model == "S0_StockOnly_G5":
            runs.append(Step7RunConfig(model=model))
        elif model == "S2_FixedSmallGate":
            for g in cfg["search"].get("fixed_gate_values", [0.10]):
                runs.append(Step7RunConfig(model=model, fixed_gate=float(g)))
        elif model == "S5_UtilityFactorizedGate":
            for p0, hidden, uw, cw, utw, cutw in product(
                cfg["search"].get("initial_gate_probabilities", [0.10]),
                cfg["search"].get("relevance_hidden_options", [[32, 16]]),
                cfg["regularization"].get("usage_weights", [0.001]),
                cfg["regularization"].get("correction_weights", [0.001]),
                cfg["utility_supervision"].get("weights", [0.05]),
                cfg["utility_supervision"].get("common_weights", [0.10]),
            ):
                runs.append(
                    Step7RunConfig(
                        model=model,
                        initial_probability=float(p0),
                        relevance_hidden=tuple(int(x) for x in hidden),
                        usage_weight=float(uw),
                        correction_weight=float(cw),
                        utility_weight=float(utw),
                        common_utility_weight=float(cutw),
                    )
                )
        else:
            for p0, hidden, uw, cw in product(
                cfg["search"].get("initial_gate_probabilities", [0.10]),
                cfg["search"].get("relevance_hidden_options", [[32, 16]]),
                cfg["regularization"].get("usage_weights", [0.001]),
                cfg["regularization"].get("correction_weights", [0.001]),
            ):
                runs.append(Step7RunConfig(model=model, initial_probability=float(p0), relevance_hidden=tuple(int(x) for x in hidden), usage_weight=float(uw), correction_weight=float(cw)))
    max_configs = cfg["search"].get("max_configs")
    if max_configs is not None:
        runs = runs[: int(max_configs)]
    max_runs = cfg.get("pilot", {}).get("max_runs")
    if bool(cfg.get("pilot", {}).get("enabled", True)) and max_runs is not None:
        runs = runs[: int(max_runs)]
    return runs


def mode_train_gate(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    if (out_dir / "gate_features.parquet").exists() and (out_dir / "feature_columns.yaml").exists() and not should_force_rebuild(cfg):
        frame = pd.read_parquet(out_dir / "gate_features.parquet")
        columns = yaml.safe_load((out_dir / "feature_columns.yaml").read_text(encoding="utf-8"))
    else:
        frame, columns = mode_build_features(cfg, logger)
    yaml.safe_dump(columns, (out_dir / "feature_columns.yaml").open("w", encoding="utf-8"), sort_keys=False)
    all_event_rows = []
    all_predictions = []
    failures = []
    folds = sorted(int(x) for x in frame["fold_id"].dropna().unique())
    for run_cfg in enumerate_run_configs(cfg):
        for seed in cfg["experiment"]["seeds"]:
            seed_everything(int(seed), bool(cfg["runtime"].get("deterministic", True)))
            for fold in folds:
                try:
                    train = frame.loc[frame["fold_id"].astype(int).ne(fold)].copy()
                    val = frame.loc[frame["fold_id"].astype(int).eq(fold)].copy()
                    ckpt_dir = Path(cfg["experiment"]["checkpoint_dir"]) / run_cfg.config_id / f"fold_{fold}" / f"seed_{seed}"
                    event_rows, state = train_gate_model(run_cfg, train, val, columns, cfg, device, ckpt_dir)
                    event_rows["fold_id"] = fold
                    event_rows["seed"] = int(seed)
                    event_rows["config_id"] = run_cfg.config_id
                    event_rows["best_epoch"] = int(state["best_epoch"])
                    pred_rows = prediction_level_from_event_rows(event_rows, cfg)
                    pred_rows["config_id"] = run_cfg.config_id
                    all_event_rows.append(event_rows)
                    all_predictions.append(pred_rows)
                    logger.info("Finished %s fold=%s seed=%s val_qlike=%.6f", run_cfg.config_id, fold, seed, pred_rows["qlike_loss"].mean())
                except Exception as exc:
                    logger.exception("Failure in %s fold=%s seed=%s", run_cfg.config_id, fold, seed)
                    failures.append({"model": run_cfg.model, "config_id": run_cfg.config_id, "fold_id": fold, "seed": seed, "message": str(exc)})
    event_df = pd.concat(all_event_rows, ignore_index=True) if all_event_rows else pd.DataFrame()
    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    atomic_parquet(event_df, out_dir / "event_stock_gate_values.parquet")
    atomic_parquet(event_df[["event_id", "date", "target_date", "source_ticker", "target_ticker", "hierarchy", "event_scope", "horizon", "split", "fold_id", "seed", "model", "is_direct_target", "static_graph_weight", "static_graph_distance", "news_correction", "gated_correction", "utility", "utility_label", "placebo_type"]], out_dir / "event_stock_corrections.parquet")
    atomic_parquet(pred_df, out_dir / "predictions_validation.parquet")
    atomic_csv(gate_diagnostics(event_df), out_dir / "utility_diagnostics.csv")
    common_detail, common_summary = common_news_impact_diagnostics(event_df)
    atomic_parquet(common_detail, out_dir / "common_news_impact.parquet")
    atomic_csv(common_summary, out_dir / "common_news_impact.csv")
    atomic_csv(pd.DataFrame(failures, columns=["model", "config_id", "fold_id", "seed", "message"]), out_dir / "failures.csv")
    if not pred_df.empty:
        tables = write_metrics(pred_df, out_dir)
        decision, reasons = decide_step7(tables["metrics_by_model"])
        write_report(out_dir, "reports/step7_stock_specific_news_report.md", decision, reasons)
        write_figures(pred_df, "results")


def mode_run_placebos(cfg: dict, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    pairs = pd.read_parquet("data/processed/step7_event_stock_pairs.parquet") if Path("data/processed/step7_event_stock_pairs.parquet").exists() else mode_build_pairs(cfg, logger)
    wrong = wrong_ticker_placebo(pairs, list(cfg["data"]["tickers"]))
    rows = [{"placebo_type": "wrong_ticker", "rows": len(wrong)}]
    atomic_csv(pd.DataFrame(rows), out_dir / "placebo_results.csv")
    logger.info("Built Step 7 pilot placebo diagnostics rows=%s", rows)


def mode_select_config(cfg: dict, logger: logging.Logger) -> dict:
    out_dir = Path(cfg["experiment"]["output_dir"])
    metrics = pd.read_csv(out_dir / "metrics_by_model.csv")
    val = metrics.loc[metrics["split"].astype(str).eq("validation")].sort_values("qlike")
    best = val.iloc[0].to_dict()
    doc = {
        "selection_rule": "minimum validation QLIKE in Step 7 pilot; locked test not used",
        "model": str(best["model"]),
        "validation_qlike": float(best["qlike"]),
        "pilot": bool(cfg.get("pilot", {}).get("enabled", True)),
    }
    yaml.safe_dump(doc, (out_dir / "best_stock_specific_gate_config.yaml").open("w", encoding="utf-8"), sort_keys=False)
    logger.info("Selected Step 7 pilot config: %s", doc)
    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 7 stock-specific news impact pilot")
    parser.add_argument("--config", default="configs/step7_stock_specific_news.yaml")
    parser.add_argument("--mode", choices=["validate-data", "reproduce-baselines", "build-events", "build-event-stock-pairs", "build-features", "train-gate", "run-placebos", "oracle-diagnostic", "select-config", "pilot", "full"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.resume:
        cfg["runtime"]["resume"] = True
    if args.force_rebuild:
        cfg["runtime"]["force_rebuild"] = True
    if args.max_runs is not None:
        cfg.setdefault("pilot", {})["max_runs"] = int(args.max_runs)
    if args.max_pairs is not None:
        cfg.setdefault("pilot", {})["max_pairs"] = int(args.max_pairs)
    if args.seeds:
        cfg["experiment"]["seeds"] = [int(x) for x in _csv(args.seeds)]
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = int(args.max_epochs)
    logger = setup_logger(cfg["experiment"]["log_dir"], args.mode)
    device = resolve_device(str(cfg["runtime"].get("device", "auto")))
    if args.mode == "pilot":
        mode_validate_data(cfg, logger)
        mode_build_features(cfg, logger)
        # The freshly rebuilt feature file should be reused by train_gate within
        # this pilot invocation instead of forcing another rebuild.
        cfg["runtime"]["force_rebuild"] = False
        mode_train_gate(cfg, device, logger)
        if bool(cfg.get("pilot", {}).get("run_placebos", False)):
            mode_run_placebos(cfg, logger)
        out_dir = Path(cfg["experiment"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_summary.json").write_text(json.dumps({"mode": args.mode, "device": str(device), "finished": True}, indent=2), encoding="utf-8")
        return
    if args.mode in {"validate-data", "pilot", "full", "reproduce-baselines"}:
        mode_validate_data(cfg, logger)
    if args.mode in {"build-events", "pilot", "full"}:
        mode_build_events(cfg, logger)
    if args.mode in {"build-event-stock-pairs", "pilot", "full"}:
        mode_build_pairs(cfg, logger)
    if args.mode in {"build-features", "pilot", "full", "oracle-diagnostic"}:
        mode_build_features(cfg, logger)
    if args.mode in {"train-gate", "pilot", "full"}:
        mode_train_gate(cfg, device, logger)
    if args.mode in {"run-placebos", "full"} or (args.mode == "pilot" and bool(cfg.get("pilot", {}).get("run_placebos", False))):
        mode_run_placebos(cfg, logger)
    if args.mode in {"select-config", "full"}:
        mode_select_config(cfg, logger)
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps({"mode": args.mode, "device": str(device), "finished": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
