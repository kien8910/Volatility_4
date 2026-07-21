from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.graph import SEMICONDUCTOR_TICKERS
from src.graph.checkpointing import load_checkpoint
from src.graph.data_loader import Step4DataError, load_config
from src.graph.datasets import GraphWindowDataset
from src.graph.panel_builder import split_indices
from src.graph.reproducibility import resolve_device, seed_everything
from src.graph.run_step4 import ModelConfig, build_model_and_adjacency, load_validated_samples, make_loader
from src.graph.scalers import StandardScaler3D
from src.news import NEWS_HIERARCHIES
from src.news.checkpointing import load_step6_checkpoint
from src.news.diagnostics import embedding_statistics, news_correction_summary
from src.news.embedding_cache import EmbeddingCache
from src.news.evaluator import (
    decide_step6,
    flatten_step6_predictions,
    news_coverage_table,
    write_step6_figures,
    write_step6_metric_tables,
    write_step6_report,
)
from src.news.news_dataset import Step6WindowDataset, ablation_mask, align_news_to_samples
from src.news.news_models import ExposedStaticBackbone, NaiveNewsFusionModel, Step6ModelConfig
from src.news.text_encoder import build_text_encoder
from src.news.text_preprocessing import (
    Step6NewsDataError,
    embedding_request_frame,
    load_or_build_news_features,
    validate_news_feature_frame,
)
from src.news.trainer import predict_step6_loader, train_step6_model, trainable_parameter_count


PRED_DEDUP_COLUMNS = ["config_id", "split", "fold_id", "seed", "date", "target_date", "ticker", "horizon"]
FAILURE_COLUMNS = ["model", "config_id", "fold_id", "seed", "message"]


def setup_logger(log_dir: str | Path, mode: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step6")
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


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def read_parquet_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def read_csv_if_exists(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=columns)


def merge_dedup(existing: pd.DataFrame, incoming: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    merged = incoming.copy() if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    usable = [col for col in subset if col in merged.columns]
    return merged.drop_duplicates(usable, keep="last").reset_index(drop=True) if usable else merged.reset_index(drop=True)


def append_predictions(pred: pd.DataFrame, out_dir: str | Path, filename: str) -> pd.DataFrame:
    out_dir = Path(out_dir)
    path = out_dir / filename
    existing = read_parquet_if_exists(path)
    merged = merge_dedup(existing, pred, PRED_DEDUP_COLUMNS)
    atomic_write_parquet(merged, path)
    return merged


def append_failure(failure: dict, out_dir: str | Path) -> None:
    path = Path(out_dir) / "failures.csv"
    existing = read_csv_if_exists(path, FAILURE_COLUMNS)
    merged = merge_dedup(existing, pd.DataFrame([failure], columns=FAILURE_COLUMNS), ["config_id", "fold_id", "seed", "message"])
    atomic_write_csv(merged, path)


def completed_run_keys(out_dir: str | Path) -> set[tuple[str, int, int]]:
    pred = read_parquet_if_exists(Path(out_dir) / "predictions_validation.parquet")
    if pred.empty:
        return set()
    return {(str(r.config_id), int(r.fold_id), int(r.seed)) for r in pred[["config_id", "fold_id", "seed"]].drop_duplicates().itertuples(index=False)}


def make_step4_cfg(cfg: dict) -> dict:
    step4_cfg_path = Path("configs/step4_static_graph.yaml")
    step4_cfg = load_config(step4_cfg_path) if step4_cfg_path.exists() else {}
    if not step4_cfg:
        raise Step4DataError("configs/step4_static_graph.yaml is required to reconstruct the Step 4 G5 backbone.")
    step4_cfg["data"].update(
        {
            "residual_state_path": cfg["data"]["residual_state_path"],
            "residual_target_path": cfg["data"]["residual_target_path"],
            "p_prediction_path": cfg["data"]["p_prediction_path"],
            "split_manifest_path": cfg["data"]["split_manifest_path"],
            "fold_manifest_path": cfg["data"]["fold_manifest_path"],
            "tickers": cfg["data"]["tickers"],
        }
    )
    step4_cfg["target"]["horizons"] = cfg["target"]["horizons"]
    step4_cfg["runtime"].update(cfg["runtime"])
    step4_cfg["training"]["batch_size"] = cfg["training"]["batch_size"]
    return step4_cfg


def parse_step4_config_id(config_id: str) -> ModelConfig:
    parts = config_id.split("__")
    if len(parts) < 5:
        raise Step4DataError(f"Cannot parse Step 4 config_id: {config_id}")
    model = parts[0]
    lookback = int(parts[1].removeprefix("L"))
    temporal = parts[2]
    loss = parts[3]
    graph_type = parts[4]
    top_k = None
    graph_embedding_dim = None
    directed = False
    graph_seed = None
    for part in parts[5:]:
        if part.startswith("k"):
            top_k = int(part[1:])
        elif part.startswith("dg"):
            graph_embedding_dim = int(part[2:])
        elif part == "directed":
            directed = True
        elif part.startswith("gseed"):
            graph_seed = int(part.removeprefix("gseed"))
    input_kind = "raw" if model in {"G0", "G4"} else "residual"
    return ModelConfig(model, input_kind, graph_type, lookback, temporal, loss, top_k, graph_embedding_dim, directed, graph_seed)


def selected_step4_config(cfg: dict) -> ModelConfig:
    path = Path(cfg["data"]["step4_config_path"])
    if not path.exists():
        raise Step4DataError(f"Missing Step 4 selected config file: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if str(doc.get("model")) != "G5":
        raise Step4DataError(f"Step 6 expects selected Step 4 model G5, found {doc.get('model')}")
    return parse_step4_config_id(str(doc["config_id"]))


def validation_fold_ids(samples) -> list[int]:
    folds = sorted(int(x) for x in np.unique(samples.fold_id[(samples.split == "validation") & (samples.fold_id >= 0)]))
    return folds or [-1]


def enumerate_step6_configs(cfg: dict) -> list[Step6ModelConfig]:
    search = cfg.get("search", {})
    models = list(search.get("include_models") or ["stock_only", "concatenation", "hierarchical_additive", "heterogeneous"])
    ablations = list(search.get("include_ablations") or ["all_dynamic", "all_including_filing"])
    poolings = list(search.get("pooling_methods") or cfg["text_encoder"].get("pooling_options", ["cls"]))
    dims = list(cfg["fusion"].get("projection_dims", [32]))
    if bool(search.get("quick_grid", False)):
        models = ["stock_only", "concatenation"]
        ablations = ["all_dynamic", "all_including_filing"]
        poolings = poolings[:1]
        dims = dims[:1]
        cfg["experiment"]["seeds"] = list(cfg["experiment"]["seeds"])[:2]
    configs: list[Step6ModelConfig] = []
    for model in models:
        usable_ablations = ["all_dynamic"] if model == "stock_only" else ablations
        usable_poolings = poolings[:1] if model == "stock_only" else poolings
        usable_dims = dims[:1] if model == "stock_only" else dims
        for ablation in usable_ablations:
            for pooling in usable_poolings:
                for dim in usable_dims:
                    configs.append(Step6ModelConfig(str(model), str(ablation), str(pooling), int(dim)))
    max_configs = search.get("max_configs")
    return configs[: int(max_configs)] if max_configs is not None else configs


def load_news_features(cfg: dict) -> pd.DataFrame:
    features = load_or_build_news_features(cfg["data"]["panel_path"], cfg["data"]["news_long_path"], list(cfg["data"]["tickers"]))
    validate_news_feature_frame(features, list(cfg["data"]["tickers"]))
    return features


def ensure_embeddings(cfg: dict, device: torch.device, logger: logging.Logger, pooling_methods: list[str] | None = None) -> pd.DataFrame:
    features = load_news_features(cfg)
    requests = embedding_request_frame(features, cfg["news"]["hierarchies"])
    cache = EmbeddingCache(cfg["text_encoder"]["cache_dir"])
    methods = pooling_methods or list(cfg["text_encoder"].get("pooling_options", ["cls"]))
    cached = cache.read()
    key_cols = ["encoder_name", "text_hash", "pooling_method", "max_length"]
    expected = {
        (str(cfg["text_encoder"]["model_name"]), str(row.text_hash), str(pooling), str(int(cfg["text_encoder"]["max_length"])))
        for pooling in methods
        for row in requests.itertuples(index=False)
    }
    if not cached.empty and key_cols[0] in cached.columns:
        existing = set(map(tuple, cached[key_cols].astype(str).to_numpy()))
        if expected.issubset(existing):
            logger.info("Step 6 embedding cache is complete for pooling=%s; encoder load skipped.", methods)
            return cached
    encoder = build_text_encoder(cfg, device)
    frame = pd.DataFrame()
    for pooling in methods:
        logger.info("Ensuring Step 6 embeddings encoder=%s pooling=%s unique_texts=%s", cfg["text_encoder"]["model_name"], pooling, len(requests))
        frame = cache.get_or_encode(
            requests,
            encoder,
            encoder_name=str(cfg["text_encoder"]["model_name"]),
            pooling_method=str(pooling),
            max_length=int(cfg["text_encoder"]["max_length"]),
        )
    return frame


def load_stock_backbone(
    cfg: dict,
    step4_cfg: dict,
    mcfg: ModelConfig,
    samples,
    train_idx: np.ndarray,
    fold_id: int | None,
    seed: int,
    final: bool,
    device: torch.device,
):
    base_model, _ = build_model_and_adjacency(mcfg, samples, train_idx, step4_cfg, seed)
    ckpt_root = Path(cfg["data"]["step4_checkpoint_dir"])
    if final:
        ckpt = ckpt_root / "final" / mcfg.config_id / f"seed_{seed}" / "best.pt"
    else:
        ckpt = ckpt_root / mcfg.config_id / f"fold_{fold_id}" / f"seed_{seed}" / "best.pt"
    if not ckpt.exists():
        raise Step4DataError(f"Missing Step 4 checkpoint required for Step 6: {ckpt}")
    state = load_checkpoint(ckpt, base_model, map_location=device)
    scaler = StandardScaler3D.from_state_dict(state["scaler_parameters"])
    return ExposedStaticBackbone(base_model), scaler, ckpt


def make_step6_model(cfg: dict, run_cfg: Step6ModelConfig, backbone: ExposedStaticBackbone, embedding_dim: int, control_dim: int) -> NaiveNewsFusionModel:
    return NaiveNewsFusionModel(
        stock_backbone=backbone,
        model=run_cfg.model,
        hierarchies=list(cfg["news"]["hierarchies"]),
        embedding_dim=embedding_dim,
        projection_dim=int(run_cfg.projection_dim),
        control_dim=control_dim,
        hidden_dims=list(cfg["fusion"]["fusion_hidden_dims"]),
        num_horizons=len(cfg["target"]["horizons"]),
        dropout=float(cfg["fusion"].get("dropout", 0.1)),
        freeze_backbone=bool(cfg["training"].get("freeze_stock_backbone", True)),
    )


def build_datasets(cfg: dict, samples, indices: dict[str, np.ndarray], scaler: StandardScaler3D, embedding_frame: pd.DataFrame, run_cfg: Step6ModelConfig):
    source = samples.residual_windows
    x_scaled = scaler.transform(source)
    features = load_news_features(cfg)
    news = align_news_to_samples(
        samples,
        features,
        embedding_frame,
        encoder_name=str(cfg["text_encoder"]["model_name"]),
        pooling_method=run_cfg.pooling_method,
        max_length=int(cfg["text_encoder"]["max_length"]),
        hierarchies=list(cfg["news"]["hierarchies"]),
    )
    mask = ablation_mask(run_cfg.ablation, news.hierarchies)
    train_base = GraphWindowDataset(samples, indices["train"], "residual", x_scaled=x_scaled)
    val_base = GraphWindowDataset(samples, indices["validation"], "residual", x_scaled=x_scaled)
    return Step6WindowDataset(train_base, news, mask), Step6WindowDataset(val_base, news, mask), news


def spike_threshold_for_indices(samples, indices: np.ndarray, quantile: float) -> float:
    return float(np.quantile(samples.target_actual[indices], quantile))


def train_or_predict_one(run_cfg: Step6ModelConfig, fold_id: int, seed: int, cfg: dict, device: torch.device, logger: logging.Logger):
    seed_everything(seed, bool(cfg["runtime"].get("deterministic", True)))
    step4_cfg = make_step4_cfg(cfg)
    mcfg = selected_step4_config(cfg)
    samples = load_validated_samples(step4_cfg, mcfg.lookback)
    idx = split_indices(samples, fold_id)
    backbone, scaler, ckpt = load_stock_backbone(cfg, step4_cfg, mcfg, samples, idx["train"], fold_id, seed, final=False, device=device)
    embedding_frame = ensure_embeddings(cfg, device, logger, [run_cfg.pooling_method])
    train_ds, val_ds, news = build_datasets(cfg, samples, idx, scaler, embedding_frame, run_cfg)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    emb_dim = int(news.embeddings.shape[-1])
    model = make_step6_model(cfg, run_cfg, backbone, emb_dim, len(news.control_columns)).to(device)
    if run_cfg.model == "stock_only":
        result_epoch = 0
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg["training"].get("scheduler_factor", 0.5)),
            patience=int(cfg["training"].get("scheduler_patience", 5)),
        )
        ckpt_dir = Path(cfg["experiment"]["checkpoint_dir"]) / run_cfg.config_id / f"fold_{fold_id}" / f"seed_{seed}"
        result = train_step6_model(
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
            scaler.state_dict(),
            resume=bool(cfg["runtime"].get("resume", False)),
            logger=logger,
        )
        load_step6_checkpoint(result.checkpoint_path, model, map_location=device)
        result_epoch = result.best_epoch
    raw = predict_step6_loader(model, val_loader, device, bool(cfg["runtime"].get("use_amp", False)), {**cfg["evaluation"], "epsilon": cfg["target"]["epsilon"]})
    pred = flatten_step6_predictions(raw, samples, news.coverage, "validation", fold_id, seed, run_cfg.as_dict(), spike_threshold_for_indices(samples, idx["train"], float(cfg["evaluation"].get("spike_quantile", 0.90))))
    pred["best_epoch"] = int(result_epoch)
    pred["step4_checkpoint"] = str(ckpt)
    trainable, total = trainable_parameter_count(model)
    params = pd.DataFrame([{**run_cfg.as_dict(), "fold_id": fold_id, "seed": seed, "trainable_parameters": trainable, "total_parameters": total}])
    return pred, news.coverage, embedding_frame, params


def mode_validate_data(cfg: dict, logger: logging.Logger) -> None:
    if list(cfg["data"]["tickers"]) != SEMICONDUCTOR_TICKERS:
        raise Step6NewsDataError(f"Step 6 ticker list must exactly match {SEMICONDUCTOR_TICKERS}")
    step4_cfg = make_step4_cfg(cfg)
    mcfg = selected_step4_config(cfg)
    samples = load_validated_samples(step4_cfg, mcfg.lookback)
    features = load_news_features(cfg)
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "news_features.parquet", index=False)
    coverage = news_coverage_table(features)
    coverage.to_csv(out_dir / "news_coverage.csv", index=False)
    logger.info("Validated Step 6 data samples=%s tickers=%s selected_step4=%s", len(samples.sample_dates), samples.tickers, mcfg.config_id)


def mode_build_embeddings(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    frame = ensure_embeddings(cfg, device, logger, list(cfg.get("search", {}).get("pooling_methods") or cfg["text_encoder"].get("pooling_options", ["cls"])))
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_statistics(frame).to_csv(out_dir / "embedding_statistics.csv", index=False)


def mode_train_validation(cfg: dict, device: torch.device, logger: logging.Logger, model_filter: set[str] | None = None) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    step4_cfg = make_step4_cfg(cfg)
    mcfg = selected_step4_config(cfg)
    samples = load_validated_samples(step4_cfg, mcfg.lookback)
    folds = validation_fold_ids(samples)
    configs = enumerate_step6_configs(cfg)
    if model_filter:
        configs = [c for c in configs if c.model in model_filter]
    done = completed_run_keys(out_dir) if bool(cfg["runtime"].get("resume", False)) else set()
    logger.info("Step 6 validation configs=%s folds=%s seeds=%s", len(configs), folds, cfg["experiment"]["seeds"])
    params_frames = []
    for run_cfg in configs:
        for fold_id in folds:
            for seed in cfg["experiment"]["seeds"]:
                key = (run_cfg.config_id, int(fold_id), int(seed))
                if key in done:
                    logger.info("Skipping completed %s fold=%s seed=%s", run_cfg.config_id, fold_id, seed)
                    continue
                start = time.time()
                try:
                    pred, coverage, embedding_frame, params = train_or_predict_one(run_cfg, int(fold_id), int(seed), cfg, device, logger)
                    pred["runtime_seconds"] = time.time() - start
                    all_pred = append_predictions(pred, out_dir, "predictions_validation.parquet")
                    atomic_write_parquet(all_pred, out_dir / "residual_predictions.parquet")
                    atomic_write_csv(news_coverage_table(coverage), out_dir / "news_coverage.csv")
                    atomic_write_csv(embedding_statistics(embedding_frame), out_dir / "embedding_statistics.csv")
                    params_frames.append(params)
                    if params_frames:
                        atomic_write_csv(pd.concat(params_frames, ignore_index=True), out_dir / "trainable_parameters.csv")
                    write_step6_metric_tables(all_pred, out_dir)
                    atomic_write_parquet(news_correction_summary(all_pred), out_dir / "news_corrections.parquet")
                    logger.info("Finished %s fold=%s seed=%s val_qlike=%.6f", run_cfg.config_id, fold_id, seed, pred["qlike_loss"].mean())
                except Exception as exc:
                    logger.exception("Failure in %s fold=%s seed=%s", run_cfg.config_id, fold_id, seed)
                    append_failure({"model": run_cfg.model, "config_id": run_cfg.config_id, "fold_id": fold_id, "seed": seed, "message": str(exc)}, out_dir)
    if not (out_dir / "failures.csv").exists():
        atomic_write_csv(pd.DataFrame(columns=FAILURE_COLUMNS), out_dir / "failures.csv")


def completed_config_ids(pred: pd.DataFrame, cfg: dict, folds: list[int]) -> set[str]:
    if pred.empty:
        return set()
    expected = len(folds) * len(cfg["experiment"]["seeds"])
    counts = pred[["config_id", "fold_id", "seed"]].drop_duplicates().groupby("config_id").size()
    return {str(config_id) for config_id, count in counts.items() if int(count) >= expected}


def mode_select_config(cfg: dict, logger: logging.Logger) -> dict:
    out_dir = Path(cfg["experiment"]["output_dir"])
    pred = read_parquet_if_exists(out_dir / "predictions_validation.parquet")
    step4_cfg = make_step4_cfg(cfg)
    samples = load_validated_samples(step4_cfg, selected_step4_config(cfg).lookback)
    complete = completed_config_ids(pred, cfg, validation_fold_ids(samples))
    if not complete:
        raise Step4DataError("No fully completed Step 6 validation configs are available.")
    scores = pred[pred["config_id"].isin(complete)].groupby(["model", "config_id", "ablation", "pooling_method"], as_index=False)["qlike_loss"].mean().sort_values("qlike_loss")
    best = scores.iloc[0]
    best_rows = pred[pred["config_id"] == best["config_id"]]
    doc = {
        "selection_rule": "minimum validation QLIKE; locked test not used",
        "model": str(best["model"]),
        "config_id": str(best["config_id"]),
        "ablation": str(best["ablation"]),
        "pooling_method": str(best["pooling_method"]),
        "validation_qlike": float(best["qlike_loss"]),
        "refit_epochs": int(max(1, best_rows["best_epoch"].median() + 1)) if "best_epoch" in best_rows else int(cfg["training"]["max_epochs"]),
        "completed_configs_considered": int(len(complete)),
    }
    yaml.safe_dump(doc, (out_dir / "best_naive_news_config.yaml").open("w", encoding="utf-8"), sort_keys=False)
    tables = write_step6_metric_tables(pred, out_dir)
    decision, reasons = decide_step6(tables["metrics_by_model"])
    write_step6_report(out_dir, "reports/step6_naive_news_report.md", decision, reasons)
    logger.info("Selected Step 6 config: %s", doc)
    return doc


def find_run_config(cfg: dict, config_id: str) -> Step6ModelConfig:
    for run_cfg in enumerate_step6_configs(cfg):
        if run_cfg.config_id == config_id:
            return run_cfg
    raise Step4DataError(f"Unknown Step 6 config_id: {config_id}")


def mode_train_final(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    if not (out_dir / "best_naive_news_config.yaml").exists():
        mode_select_config(cfg, logger)
    best = yaml.safe_load((out_dir / "best_naive_news_config.yaml").read_text(encoding="utf-8"))
    run_cfg = find_run_config(cfg, best["config_id"])
    final_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    final_cfg["training"]["max_epochs"] = int(best["refit_epochs"])
    step4_cfg = make_step4_cfg(final_cfg)
    mcfg = selected_step4_config(final_cfg)
    samples = load_validated_samples(step4_cfg, mcfg.lookback)
    idx = split_indices(samples, None)
    for seed in final_cfg["experiment"]["seeds"]:
        seed_everything(int(seed), bool(final_cfg["runtime"].get("deterministic", True)))
        backbone, scaler, _ = load_stock_backbone(final_cfg, step4_cfg, mcfg, samples, idx["development"], None, int(seed), final=True, device=device)
        embedding_frame = ensure_embeddings(final_cfg, device, logger, [run_cfg.pooling_method])
        train_ds, val_ds, news = build_datasets(final_cfg, samples, {"train": idx["development"], "validation": idx["development"]}, scaler, embedding_frame, run_cfg)
        loader = make_loader(train_ds, final_cfg, shuffle=True)
        model = make_step6_model(final_cfg, run_cfg, backbone, int(news.embeddings.shape[-1]), len(news.control_columns)).to(device)
        if run_cfg.model == "stock_only":
            continue
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(final_cfg["training"]["learning_rate"]), weight_decay=float(final_cfg["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        ckpt_dir = Path(final_cfg["experiment"]["checkpoint_dir"]) / "final" / run_cfg.config_id / f"seed_{seed}"
        train_step6_model(model, loader, loader, optimizer, scheduler, device, final_cfg, ckpt_dir, run_cfg.as_dict(), int(seed), samples.tickers, scaler.state_dict(), resume=bool(final_cfg["runtime"].get("resume", False)), logger=logger)


def mode_evaluate_test(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    best = yaml.safe_load((out_dir / "best_naive_news_config.yaml").read_text(encoding="utf-8"))
    run_cfg = find_run_config(cfg, best["config_id"])
    step4_cfg = make_step4_cfg(cfg)
    mcfg = selected_step4_config(cfg)
    samples = load_validated_samples(step4_cfg, mcfg.lookback)
    idx = split_indices(samples, None)
    frames = []
    for seed in cfg["experiment"]["seeds"]:
        backbone, scaler, _ = load_stock_backbone(cfg, step4_cfg, mcfg, samples, idx["development"], None, int(seed), final=True, device=device)
        embedding_frame = ensure_embeddings(cfg, device, logger, [run_cfg.pooling_method])
        _, test_ds, news = build_datasets(cfg, samples, {"train": idx["development"], "validation": idx["test"]}, scaler, embedding_frame, run_cfg)
        loader = make_loader(test_ds, cfg, shuffle=False)
        model = make_step6_model(cfg, run_cfg, backbone, int(news.embeddings.shape[-1]), len(news.control_columns)).to(device)
        if run_cfg.model != "stock_only":
            ckpt = Path(cfg["experiment"]["checkpoint_dir"]) / "final" / run_cfg.config_id / f"seed_{seed}" / "best.pt"
            load_step6_checkpoint(ckpt, model, map_location=device)
        raw = predict_step6_loader(model, loader, device, bool(cfg["runtime"].get("use_amp", False)), {**cfg["evaluation"], "epsilon": cfg["target"]["epsilon"]})
        frames.append(flatten_step6_predictions(raw, samples, news.coverage, "test", -1, int(seed), run_cfg.as_dict(), spike_threshold_for_indices(samples, idx["development"], float(cfg["evaluation"].get("spike_quantile", 0.90)))))
    test_pred = pd.concat(frames, ignore_index=True)
    atomic_write_parquet(test_pred, out_dir / "predictions_test.parquet")
    val = read_parquet_if_exists(out_dir / "predictions_validation.parquet")
    all_pred = pd.concat([val, test_pred], ignore_index=True) if not val.empty else test_pred
    atomic_write_parquet(all_pred, out_dir / "residual_predictions.parquet")
    atomic_write_parquet(news_correction_summary(all_pred), out_dir / "news_corrections.parquet")
    tables = write_step6_metric_tables(all_pred, out_dir)
    decision, reasons = decide_step6(tables["metrics_by_model"])
    write_step6_report(out_dir, "reports/step6_naive_news_report.md", decision, reasons)
    write_step6_figures(all_pred, "results")
    logger.info("Step 6 test evaluation complete decision=%s", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6 naive hierarchical news fusion")
    parser.add_argument("--config", default="configs/step6_naive_news.yaml")
    parser.add_argument("--mode", choices=["validate-data", "reproduce-step4", "build-embeddings", "train-validation", "select-config", "train-final", "evaluate-test", "full"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-models", default=None)
    parser.add_argument("--include-ablations", default=None)
    parser.add_argument("--pooling-methods", default=None)
    parser.add_argument("--projection-dims", default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--quick-grid", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--text-model", default=None)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
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
    if args.include_models:
        cfg["search"]["include_models"] = parse_csv(args.include_models)
    if args.include_ablations:
        cfg["search"]["include_ablations"] = parse_csv(args.include_ablations)
    if args.pooling_methods:
        cfg["search"]["pooling_methods"] = parse_csv(args.pooling_methods)
    if args.projection_dims:
        cfg["fusion"]["projection_dims"] = [int(x) for x in parse_csv(args.projection_dims)]
    if args.max_configs is not None:
        cfg["search"]["max_configs"] = int(args.max_configs)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        cfg["training"]["early_stopping_patience"] = int(args.patience)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.quick_grid:
        cfg["search"]["quick_grid"] = True
    if args.seeds:
        cfg["experiment"]["seeds"] = [int(x) for x in parse_csv(args.seeds)]
    if args.text_model:
        cfg["text_encoder"]["model_name"] = args.text_model
    logger = setup_logger(cfg["experiment"]["log_dir"], args.mode)
    device = resolve_device(str(cfg["runtime"].get("device", "auto")))
    logger.info("Step 6 mode=%s device=%s text_model=%s", args.mode, device, cfg["text_encoder"]["model_name"])
    if args.mode in {"validate-data", "full"}:
        mode_validate_data(cfg, logger)
    if args.mode in {"build-embeddings", "full"}:
        mode_build_embeddings(cfg, device, logger)
    if args.mode in {"reproduce-step4"}:
        mode_train_validation(cfg, device, logger, model_filter={"stock_only"})
    if args.mode in {"train-validation", "full"}:
        mode_train_validation(cfg, device, logger)
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
