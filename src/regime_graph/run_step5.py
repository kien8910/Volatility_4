from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import torch
import yaml
from torch.utils.data import DataLoader

from src.graph.data_loader import Step4DataError, load_config, load_inputs, validate_inputs
from src.graph.datasets import GraphWindowDataset
from src.graph.panel_builder import build_panels, build_sample_table, split_indices
from src.graph.reproducibility import resolve_device, seed_everything
from src.graph.scalers import StandardScaler3D
from src.regime_graph.diagnostics import graph_diversity_from_bank, graph_edges_from_bank, regime_usage_from_predictions
from src.regime_graph.evaluator import (
    decide_step5,
    flatten_step5_predictions,
    write_step5_figures,
    write_step5_metric_tables,
    write_step5_report,
)
from src.regime_graph.regime_graph_model import RegimeGraphForecastModel
from src.regime_graph.state_features import (
    StateFeatureScaler,
    build_state_feature_frame,
    market_state_labels,
    state_feature_matrix,
)
from src.regime_graph.trainer import Step5WindowDataset, predict_step5_loader, train_step5_model
from src.regime_graph.checkpointing import load_step5_checkpoint


@dataclass(frozen=True)
class Step5RunConfig:
    model: str
    K: int
    ema_beta: float
    gate_temperature: float
    gate_hidden_dim: int
    gate_regularization: str
    gate_regularization_weight: float
    loss: str = "mse"

    @property
    def config_id(self) -> str:
        return (
            f"{self.model}__K{self.K}__beta{self.ema_beta:g}"
            f"__tau{self.gate_temperature:g}__gh{self.gate_hidden_dim}"
            f"__reg{self.gate_regularization}__rw{self.gate_regularization_weight:g}"
        )

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "K": self.K,
            "ema_beta": self.ema_beta,
            "gate_temperature": self.gate_temperature,
            "gate_hidden_dim": self.gate_hidden_dim,
            "gate_regularization": self.gate_regularization,
            "gate_regularization_weight": self.gate_regularization_weight,
            "loss": self.loss,
            "config_id": self.config_id,
        }


PRED_DEDUP_COLUMNS = ["config_id", "split", "fold_id", "seed", "date", "target_date", "ticker", "horizon"]
EDGE_DEDUP_COLUMNS = ["config_id", "fold_id", "seed", "regime", "source", "target"]
FAILURE_COLUMNS = ["model", "config_id", "fold_id", "seed", "message"]
GRAPH_DIVERSITY_COLUMNS = [
    "config_id",
    "model",
    "fold_id",
    "seed",
    "regime_a",
    "regime_b",
    "frobenius_distance",
    "cosine_similarity",
    "spearman_correlation",
    "topk_jaccard",
]


def setup_logger(log_dir: str | Path, mode: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(Path(log_dir) / f"{mode}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def parse_step4_backbone(config_path: str | Path, fallback: dict) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise Step4DataError(f"Missing Step 4 selected config: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    config_id = str(doc["config_id"])
    parts = config_id.split("__")
    parsed = {
        "model": doc.get("model", "G5"),
        "config_id": config_id,
        "lookback": int(parts[1].lstrip("L")),
        "temporal_encoder": parts[2],
        "loss": parts[3],
        "graph_type": parts[4],
        "top_k": int(next(item[1:] for item in parts if item.startswith("k"))),
        "graph_embedding_dim": int(next(item[2:] for item in parts if item.startswith("dg"))),
        "directed": "directed" in parts,
        "validation_qlike": float(doc.get("validation_qlike", np.nan)),
    }
    if parsed["model"] != "G5" or parsed["temporal_encoder"] != "small_tcn" or parsed["graph_type"] != "learned":
        raise Step4DataError(f"Step 5 expects locked Step 4 G5 small_tcn learned graph. Found: {config_id}")
    merged = {**fallback, **parsed}
    return merged


def load_samples_and_features(cfg: dict):
    backbone = parse_step4_backbone(cfg["data"]["step4_config_path"], cfg["backbone"])
    inputs = validate_inputs(load_inputs(cfg), list(cfg["data"]["tickers"]), list(backbone["horizons"]))
    panels = build_panels(inputs.residual_state, inputs.residual_targets, cfg["data"]["tickers"], list(backbone["horizons"]))
    samples = build_sample_table(panels, int(backbone["lookback"]))
    state_frame = build_state_feature_frame(samples)
    return samples, state_frame, backbone


def validation_fold_ids(samples) -> list[int]:
    folds = sorted(int(x) for x in np.unique(samples.fold_id[(samples.split == "validation") & (samples.fold_id >= 0)]))
    return folds or [-1]


def enumerate_step5_configs(cfg: dict) -> list[Step5RunConfig]:
    include = set(cfg["search"].get("include_models") or ["S5-B0", "S5-E", "S5-R"])
    configs: list[Step5RunConfig] = []
    base_gate_hidden = int(cfg["gate"]["hidden_dims"][0])
    base_tau = float(cfg["gate"]["temperatures"][0])
    if "S5-B0" in include:
        configs.append(Step5RunConfig("S5-B0", 1, 0.0, base_tau, base_gate_hidden, "none", 0.0, cfg["training"].get("loss", "mse")))
    if "S5-E" in include:
        for beta in cfg["ema"]["betas"]:
            configs.append(Step5RunConfig("S5-E", 1, float(beta), base_tau, base_gate_hidden, "none", 0.0, cfg["training"].get("loss", "mse")))
    if "S5-R" in include:
        for K, tau, hidden, reg, rw in product(
            [k for k in cfg["graph_bank"]["K_values"] if int(k) > 1],
            cfg["gate"]["temperatures"],
            cfg["gate"]["hidden_dims"],
            cfg["gate"]["regularization_options"],
            cfg["gate"]["regularization_weights"],
        ):
            configs.append(Step5RunConfig("S5-R", int(K), 0.0, float(tau), int(hidden), str(reg), float(rw), cfg["training"].get("loss", "mse")))
    if "S5-RE" in include:
        for K, beta, tau, hidden, reg, rw in product(
            [k for k in cfg["graph_bank"]["K_values"] if int(k) > 1],
            [b for b in cfg["ema"]["betas"] if float(b) > 0],
            cfg["gate"]["temperatures"],
            cfg["gate"]["hidden_dims"],
            cfg["gate"]["regularization_options"],
            cfg["gate"]["regularization_weights"],
        ):
            configs.append(Step5RunConfig("S5-RE", int(K), float(beta), float(tau), int(hidden), str(reg), float(rw), cfg["training"].get("loss", "mse")))
    max_configs = cfg["search"].get("max_configs")
    return configs[: int(max_configs)] if max_configs is not None else configs


def make_loader(dataset, cfg: dict, shuffle: bool) -> DataLoader:
    device = resolve_device(str(cfg["runtime"].get("device", "auto")))
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["runtime"].get("num_workers", 0)),
        pin_memory=bool(cfg["runtime"].get("pin_memory", False)) and device.type == "cuda",
    )


def _atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)


def merge_dedup(existing: pd.DataFrame, incoming: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    merged = incoming.copy() if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    usable = [col for col in subset if col in merged.columns]
    return merged.drop_duplicates(usable, keep="last").reset_index(drop=True) if usable else merged.reset_index(drop=True)


def append_table(df: pd.DataFrame, path: str | Path, subset: list[str], parquet: bool, columns: list[str] | None = None) -> pd.DataFrame:
    if df.empty and len(df.columns) == 0:
        existing = _read_parquet(path) if parquet else _read_csv(path, columns=columns)
        if not Path(path).exists() and columns is not None:
            empty = pd.DataFrame(columns=columns)
            if parquet:
                _atomic_write_parquet(empty, path)
            else:
                _atomic_write_csv(empty, path)
            return empty
        return existing
    existing = _read_parquet(path) if parquet else _read_csv(path, columns=columns)
    merged = merge_dedup(existing, df, subset)
    if parquet:
        _atomic_write_parquet(merged, path)
    else:
        _atomic_write_csv(merged, path)
    return merged


def completed_run_keys(out_dir: str | Path) -> set[tuple[str, int, int]]:
    pred = _read_parquet(Path(out_dir) / "predictions_validation.parquet")
    if pred.empty:
        return set()
    keys = pred[["config_id", "fold_id", "seed"]].drop_duplicates()
    return {(str(r.config_id), int(r.fold_id), int(r.seed)) for r in keys.itertuples(index=False)}


def build_model(run_cfg: Step5RunConfig, backbone: dict, cfg: dict, state_dim: int) -> RegimeGraphForecastModel:
    temporal_cfg = {
        "hidden_dim": int(backbone["hidden_dim"]),
        "channels": list(backbone["channels"]),
        "kernel_size": int(backbone["kernel_size"]),
        "dropout": float(backbone["dropout"]),
        "activation": str(backbone["activation"]),
    }
    return RegimeGraphForecastModel(
        num_nodes=len(cfg["data"]["tickers"]),
        lookback=int(backbone["lookback"]),
        num_horizons=len(backbone["horizons"]),
        temporal_cfg=temporal_cfg,
        graph_embedding_dim=int(backbone["graph_embedding_dim"]),
        top_k=int(backbone["top_k"]),
        directed=bool(backbone["directed"]),
        num_graphs=int(run_cfg.K),
        model_id=run_cfg.model,
        ema_beta=float(run_cfg.ema_beta),
        state_dim=state_dim,
        gate_hidden_dim=int(run_cfg.gate_hidden_dim),
        gate_temperature=float(run_cfg.gate_temperature),
        gate_dropout=float(cfg["gate"].get("dropout", 0.1)),
        use_ema_for_training=bool(cfg["ema"].get("use_for_training", False)),
        use_ema_for_validation=bool(cfg["ema"].get("use_for_validation", True)),
    )


def train_one(run_cfg: Step5RunConfig, fold_id: int, seed: int, cfg: dict, device: torch.device, logger: logging.Logger):
    seed_everything(seed, deterministic=bool(cfg["runtime"].get("deterministic", True)))
    samples, state_frame, backbone = load_samples_and_features(cfg)
    idx = split_indices(samples, fold_id)
    source_scaler = StandardScaler3D().fit(samples.residual_windows[idx["train"]])
    x_scaled = source_scaler.transform(samples.residual_windows)
    state_values = state_feature_matrix(state_frame, list(cfg["state_features"]["include"]))
    state_scaler = StateFeatureScaler().fit(state_values[idx["train"]])
    state_scaled = state_scaler.transform(state_values)
    q_low, q_high = cfg["evaluation"]["market_state_quantiles"]
    market_states = market_state_labels(state_frame, idx["train"], float(q_low), float(q_high))
    train_base = GraphWindowDataset(samples, idx["train"], input_kind="residual", x_scaled=x_scaled)
    val_base = GraphWindowDataset(samples, idx["validation"], input_kind="residual", x_scaled=x_scaled)
    train_ds = Step5WindowDataset(train_base, state_scaled, market_states)
    val_ds = Step5WindowDataset(val_base, state_scaled, market_states)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    model = build_model(run_cfg, backbone, cfg, state_dim=state_scaled.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["training"].get("scheduler_factor", 0.5)),
        patience=int(cfg["training"].get("scheduler_patience", 5)),
    )
    ckpt_dir = Path(cfg["experiment"]["checkpoint_dir"]) / run_cfg.model / f"K_{run_cfg.K}" / f"beta_{run_cfg.ema_beta:g}" / f"fold_{fold_id}" / f"seed_{seed}"
    result = train_step5_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        cfg,
        ckpt_dir,
        run_cfg.as_dict(),
        seed,
        samples.tickers,
        source_scaler.state_dict(),
        state_scaler.state_dict(),
        resume=bool(cfg["runtime"].get("resume", False)),
        logger=logger,
    )
    load_step5_checkpoint(result.checkpoint_path, model, map_location=device)
    raw = predict_step5_loader(model, val_loader, device, bool(cfg["runtime"].get("use_amp", False)), cfg["evaluation"])
    pred = flatten_step5_predictions(raw, samples, samples.tickers, samples.horizons, "validation", fold_id, seed, run_cfg.as_dict(), float(cfg["evaluation"]["spike_quantile"]))
    pred["best_epoch"] = result.best_epoch
    adj = model.adjacency_bank()
    edges = graph_edges_from_bank(adj, samples.tickers, run_cfg.as_dict(), fold_id, seed)
    diversity = graph_diversity_from_bank(adj, run_cfg.as_dict(), fold_id, seed)
    state_frame = state_frame.copy()
    state_frame["scaler_split_used"] = "train"
    return pred, edges, diversity, state_frame


def mode_validate_data(cfg: dict, logger: logging.Logger) -> None:
    samples, state_frame, backbone = load_samples_and_features(cfg)
    logger.info("Validated Step 5 data samples=%s backbone=%s", len(samples.sample_dates), backbone["config_id"])
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    state_frame.to_parquet(out_dir / "state_features.parquet", index=False)


def mode_train_validation(cfg: dict, device: torch.device, logger: logging.Logger, model_filter: set[str] | None = None) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, _, _ = load_samples_and_features(cfg)
    folds = validation_fold_ids(samples)
    configs = enumerate_step5_configs(cfg)
    if model_filter:
        configs = [c for c in configs if c.model in model_filter]
    done = completed_run_keys(out_dir) if bool(cfg["runtime"].get("resume", False)) else set()
    logger.info("Step 5 validation configs=%s folds=%s seeds=%s", len(configs), folds, cfg["experiment"]["seeds"])
    for run_cfg in configs:
        for fold_id in folds:
            for seed in cfg["experiment"]["seeds"]:
                key = (run_cfg.config_id, int(fold_id), int(seed))
                if key in done:
                    logger.info("Skipping completed %s fold=%s seed=%s", run_cfg.config_id, fold_id, seed)
                    continue
                start = time.time()
                try:
                    pred, edges, diversity, state_frame = train_one(run_cfg, int(fold_id), int(seed), cfg, device, logger)
                    pred["runtime_seconds"] = time.time() - start
                    all_pred = append_table(pred, out_dir / "predictions_validation.parquet", PRED_DEDUP_COLUMNS, parquet=True)
                    append_table(all_pred, out_dir / "residual_predictions.parquet", PRED_DEDUP_COLUMNS, parquet=True)
                    append_table(edges, out_dir / "graph_edges.csv", EDGE_DEDUP_COLUMNS, parquet=False)
                    append_table(diversity, out_dir / "graph_diversity.csv", ["config_id", "fold_id", "seed", "regime_a", "regime_b"], parquet=False, columns=GRAPH_DIVERSITY_COLUMNS)
                    state_frame.to_parquet(out_dir / "state_features.parquet", index=False)
                    write_step5_metric_tables(all_pred, out_dir)
                    regime_usage_from_predictions(all_pred).to_csv(out_dir / "regime_usage.csv", index=False)
                    regime_usage_from_predictions(all_pred).to_csv(out_dir / "regime_entropy.csv", index=False)
                    pd.DataFrame(columns=["config_id", "model", "fold_id", "seed", "mean_miad", "median_miad", "last10_miad", "max_miad"]).to_csv(out_dir / "ema_stability.csv", index=False)
                    logger.info("Finished %s fold=%s seed=%s val_qlike=%.6f", run_cfg.config_id, fold_id, seed, pred["qlike_loss"].mean())
                except Exception as exc:
                    logger.exception("Failure in %s fold=%s seed=%s", run_cfg.config_id, fold_id, seed)
                    failure = pd.DataFrame(
                        [{"model": run_cfg.model, "config_id": run_cfg.config_id, "fold_id": fold_id, "seed": seed, "message": str(exc)}],
                        columns=FAILURE_COLUMNS,
                    )
                    append_table(failure, out_dir / "failures.csv", ["config_id", "fold_id", "seed", "message"], parquet=False)
    if not (out_dir / "failures.csv").exists():
        pd.DataFrame(columns=FAILURE_COLUMNS).to_csv(out_dir / "failures.csv", index=False)


def completed_config_ids(pred: pd.DataFrame, cfg: dict, folds: list[int]) -> set[str]:
    if pred.empty:
        return set()
    run_counts = pred[["config_id", "fold_id", "seed"]].drop_duplicates().groupby("config_id").size()
    expected = len(folds) * len(cfg["experiment"]["seeds"])
    return {str(config_id) for config_id, count in run_counts.items() if int(count) >= expected}


def mode_select_config(cfg: dict, logger: logging.Logger) -> dict:
    out_dir = Path(cfg["experiment"]["output_dir"])
    pred = _read_parquet(out_dir / "predictions_validation.parquet")
    samples, _, _ = load_samples_and_features(cfg)
    complete = completed_config_ids(pred, cfg, validation_fold_ids(samples))
    if not complete:
        raise Step4DataError("No fully completed Step 5 validation configs are available.")
    chosen = pred[pred["config_id"].isin(complete)].groupby(["model", "config_id"], as_index=False)["qlike_loss"].mean().sort_values("qlike_loss").iloc[0]
    best_rows = pred[pred["config_id"] == chosen["config_id"]]
    best_doc = {
        "selection_rule": "minimum validation QLIKE; locked test not used",
        "model": str(chosen["model"]),
        "config_id": str(chosen["config_id"]),
        "validation_qlike": float(chosen["qlike_loss"]),
        "refit_epochs": int(max(1, best_rows["best_epoch"].median() + 1)) if "best_epoch" in best_rows else int(cfg["training"]["max_epochs"]),
        "completed_configs_considered": int(len(complete)),
    }
    yaml.safe_dump(best_doc, (out_dir / "best_step5_config.yaml").open("w", encoding="utf-8"), sort_keys=False)
    logger.info("Selected Step 5 config: %s", best_doc)
    return best_doc


def _run_config_from_id(configs: list[Step5RunConfig], config_id: str) -> Step5RunConfig:
    for cfg in configs:
        if cfg.config_id == config_id:
            return cfg
    raise Step4DataError(f"Unknown Step 5 config_id: {config_id}")


def mode_train_final(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    if not (out_dir / "best_step5_config.yaml").exists():
        mode_select_config(cfg, logger)
    best = yaml.safe_load((out_dir / "best_step5_config.yaml").read_text(encoding="utf-8"))
    run_cfg = _run_config_from_id(enumerate_step5_configs(cfg), best["config_id"])
    final_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    final_cfg["training"]["max_epochs"] = int(best["refit_epochs"])
    samples, state_frame, backbone = load_samples_and_features(final_cfg)
    idx = split_indices(samples, None)
    for seed in final_cfg["experiment"]["seeds"]:
        seed_everything(int(seed), bool(final_cfg["runtime"].get("deterministic", True)))
        source_scaler = StandardScaler3D().fit(samples.residual_windows[idx["development"]])
        x_scaled = source_scaler.transform(samples.residual_windows)
        state_values = state_feature_matrix(state_frame, list(final_cfg["state_features"]["include"]))
        state_scaler = StateFeatureScaler().fit(state_values[idx["development"]])
        state_scaled = state_scaler.transform(state_values)
        q_low, q_high = final_cfg["evaluation"]["market_state_quantiles"]
        market_states = market_state_labels(state_frame, idx["development"], float(q_low), float(q_high))
        dev_base = GraphWindowDataset(samples, idx["development"], input_kind="residual", x_scaled=x_scaled)
        dev_ds = Step5WindowDataset(dev_base, state_scaled, market_states)
        dev_loader = make_loader(dev_ds, final_cfg, shuffle=True)
        model = build_model(run_cfg, backbone, final_cfg, state_dim=state_scaled.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(final_cfg["training"]["learning_rate"]), weight_decay=float(final_cfg["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        ckpt_dir = Path(final_cfg["experiment"]["checkpoint_dir"]) / "final" / run_cfg.model / f"K_{run_cfg.K}" / f"beta_{run_cfg.ema_beta:g}" / f"seed_{seed}"
        train_step5_model(model, dev_loader, dev_loader, optimizer, scheduler, device, final_cfg, ckpt_dir, run_cfg.as_dict(), int(seed), samples.tickers, source_scaler.state_dict(), state_scaler.state_dict(), resume=bool(final_cfg["runtime"].get("resume", False)), logger=logger)


def mode_evaluate_test(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    best = yaml.safe_load((out_dir / "best_step5_config.yaml").read_text(encoding="utf-8"))
    run_cfg = _run_config_from_id(enumerate_step5_configs(cfg), best["config_id"])
    samples, state_frame, backbone = load_samples_and_features(cfg)
    idx = split_indices(samples, None)
    frames = []
    for seed in cfg["experiment"]["seeds"]:
        source_scaler = StandardScaler3D().fit(samples.residual_windows[idx["development"]])
        x_scaled = source_scaler.transform(samples.residual_windows)
        state_values = state_feature_matrix(state_frame, list(cfg["state_features"]["include"]))
        state_scaler = StateFeatureScaler().fit(state_values[idx["development"]])
        state_scaled = state_scaler.transform(state_values)
        q_low, q_high = cfg["evaluation"]["market_state_quantiles"]
        market_states = market_state_labels(state_frame, idx["development"], float(q_low), float(q_high))
        test_base = GraphWindowDataset(samples, idx["test"], input_kind="residual", x_scaled=x_scaled)
        test_ds = Step5WindowDataset(test_base, state_scaled, market_states)
        test_loader = make_loader(test_ds, cfg, shuffle=False)
        model = build_model(run_cfg, backbone, cfg, state_dim=state_scaled.shape[1]).to(device)
        ckpt = Path(cfg["experiment"]["checkpoint_dir"]) / "final" / run_cfg.model / f"K_{run_cfg.K}" / f"beta_{run_cfg.ema_beta:g}" / f"seed_{seed}" / "best.pt"
        load_step5_checkpoint(ckpt, model, map_location=device)
        raw = predict_step5_loader(model, test_loader, device, bool(cfg["runtime"].get("use_amp", False)), cfg["evaluation"])
        frames.append(flatten_step5_predictions(raw, samples, samples.tickers, samples.horizons, "test", -1, int(seed), run_cfg.as_dict(), float(cfg["evaluation"]["spike_quantile"])))
    test_pred = pd.concat(frames, ignore_index=True)
    _atomic_write_parquet(test_pred, out_dir / "predictions_test.parquet")
    val = _read_parquet(out_dir / "predictions_validation.parquet")
    all_pred = pd.concat([val, test_pred], ignore_index=True)
    _atomic_write_parquet(all_pred, out_dir / "residual_predictions.parquet")
    tables = write_step5_metric_tables(all_pred, out_dir)
    decision, reasons = decide_step5(tables["metrics_by_model"])
    write_step5_report(out_dir, "reports/step5_regime_graph_report.md", decision, reasons)
    write_step5_figures(all_pred, "results")
    logger.info("Step 5 test evaluation complete decision=%s", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5 EMA/regime graph learning")
    parser.add_argument("--config", default="configs/step5_regime_graph.yaml")
    parser.add_argument("--mode", choices=["validate-data", "reproduce-step4", "train-validation", "train-ema", "train-regime", "train-combined", "select-config", "train-final", "evaluate-test", "full"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-models", default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = args.num_workers
    if args.resume:
        cfg["runtime"]["resume"] = True
    if args.include_models is not None:
        cfg["search"]["include_models"] = _parse_csv(args.include_models)
    if args.max_configs is not None:
        cfg["search"]["max_configs"] = int(args.max_configs)
    if args.seeds is not None:
        cfg["experiment"]["seeds"] = [int(item) for item in _parse_csv(args.seeds)]
    logger = setup_logger(cfg["experiment"]["log_dir"], args.mode)
    device = resolve_device(str(cfg["runtime"].get("device", "auto")))
    logger.info("Step 5 mode=%s device=%s include_models=%s", args.mode, device, cfg["search"].get("include_models"))
    if args.mode in {"validate-data", "full"}:
        mode_validate_data(cfg, logger)
    if args.mode in {"reproduce-step4", "full"}:
        mode_train_validation(cfg, device, logger, model_filter={"S5-B0"})
    if args.mode in {"train-validation", "full"}:
        mode_train_validation(cfg, device, logger)
    if args.mode == "train-ema":
        mode_train_validation(cfg, device, logger, model_filter={"S5-B0", "S5-E"})
    if args.mode == "train-regime":
        mode_train_validation(cfg, device, logger, model_filter={"S5-B0", "S5-R"})
    if args.mode == "train-combined":
        mode_train_validation(cfg, device, logger, model_filter={"S5-B0", "S5-RE"})
    if args.mode in {"select-config", "full"}:
        mode_select_config(cfg, logger)
    if args.mode in {"train-final", "full"}:
        mode_train_final(cfg, device, logger)
    if args.mode in {"evaluate-test", "full"}:
        mode_evaluate_test(cfg, device, logger)
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps({"mode": args.mode, "device": str(device), "finished": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
