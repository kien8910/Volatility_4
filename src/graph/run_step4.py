from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.graph.adjacency import AdjacencySpec, build_fixed_adjacency, correlation_adjacency, random_adjacency
from src.graph.data_loader import Step4DataError, load_config, load_inputs, validate_inputs
from src.graph.datasets import GraphWindowDataset
from src.graph.evaluator import decide_go_no_go, flatten_predictions, write_figures, write_metric_tables, write_report
from src.graph.graph_diagnostics import (
    adjacency_to_edges,
    graph_stability,
    mean_adjacency_from_edges,
    plot_graph,
    plot_graph_stability,
    plot_unavailable_graph,
)
from src.graph.masked_reconstruction import ReconstructionConfig, run_reconstruction_diagnostic
from src.graph.models import StaticGraphForecastModel
from src.graph.panel_builder import build_panels, build_sample_table, split_indices
from src.graph.reproducibility import resolve_device, seed_everything
from src.graph.scalers import StandardScaler3D
from src.graph.trainer import predict_loader, train_model
from src.graph.checkpointing import load_checkpoint


@dataclass(frozen=True)
class ModelConfig:
    model: str
    input_kind: str
    graph_type: str
    lookback: int
    temporal: str
    loss: str
    top_k: int | None = None
    graph_embedding_dim: int | None = None
    directed: bool = False
    graph_seed: int | None = None

    @property
    def config_id(self) -> str:
        parts = [self.model, f"L{self.lookback}", self.temporal, self.loss, self.graph_type]
        if self.top_k is not None:
            parts.append(f"k{self.top_k}")
        if self.graph_embedding_dim is not None:
            parts.append(f"dg{self.graph_embedding_dim}")
        if self.directed:
            parts.append("directed")
        if self.graph_seed is not None:
            parts.append(f"gseed{self.graph_seed}")
        return "__".join(parts)


def setup_logger(log_dir: str | Path, mode: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step4")
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


def enumerate_configs(cfg: dict) -> list[ModelConfig]:
    lookbacks = list(cfg["window"]["lookbacks"])
    temporals = list(cfg["temporal_encoder"]["options"])
    losses = list(cfg["training"]["loss_options"])
    top_ks = list(cfg["graph"]["top_k"])
    dims = list(cfg["graph"]["embedding_dims"])
    directed_options = list(cfg["graph"]["directed_options"])
    graph_seeds = list(cfg["experiment"]["seeds"])
    configs: list[ModelConfig] = []
    for lookback, temporal, loss in product(lookbacks, temporals, losses):
        configs.append(ModelConfig("G0", "raw", "none", lookback, temporal, loss))
        configs.append(ModelConfig("G1", "residual", "identity", lookback, temporal, loss))
        for top_k in top_ks:
            configs.append(ModelConfig("G2", "residual", "correlation", lookback, temporal, loss, top_k=top_k))
            for graph_seed in graph_seeds:
                configs.append(ModelConfig("G3", "residual", "random", lookback, temporal, loss, top_k=top_k, graph_seed=graph_seed))
        for top_k, dim, directed in product(top_ks, dims, directed_options):
            configs.append(
                ModelConfig("G4", "raw", "learned", lookback, temporal, loss, top_k=top_k, graph_embedding_dim=dim, directed=directed)
            )
            configs.append(
                ModelConfig(
                    "G5",
                    "residual",
                    "learned",
                    lookback,
                    temporal,
                    loss,
                    top_k=top_k,
                    graph_embedding_dim=dim,
                    directed=directed,
                )
            )
    return configs


def load_validated_samples(cfg: dict, lookback: int):
    inputs = validate_inputs(load_inputs(cfg), list(cfg["data"]["tickers"]), list(cfg["target"]["horizons"]))
    panels = build_panels(inputs.residual_state, inputs.residual_targets, cfg["data"]["tickers"], cfg["target"]["horizons"])
    return build_sample_table(panels, lookback)


def make_loader(dataset, cfg: dict, shuffle: bool) -> DataLoader:
    runtime = cfg["runtime"]
    device = resolve_device(str(runtime.get("device", "auto")))
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=bool(runtime.get("pin_memory", False)) and device.type == "cuda",
    )


def build_model_and_adjacency(mcfg: ModelConfig, samples, train_idx: np.ndarray, cfg: dict, seed: int):
    source = samples.residual_windows if mcfg.input_kind == "residual" else samples.raw_windows
    fixed = build_fixed_adjacency(
        AdjacencySpec(
            graph_type=mcfg.graph_type,
            top_k=mcfg.top_k,
            directed=mcfg.directed,
            graph_seed=mcfg.graph_seed or seed,
        ),
        train_windows=source[train_idx],
        num_nodes=len(samples.tickers),
    )
    model = StaticGraphForecastModel(
        num_nodes=len(samples.tickers),
        lookback=mcfg.lookback,
        num_horizons=len(samples.horizons),
        temporal_kind=mcfg.temporal,
        temporal_cfg=cfg["temporal_encoder"],
        graph_type=mcfg.graph_type,
        fixed_adjacency=fixed,
        graph_embedding_dim=mcfg.graph_embedding_dim,
        top_k=mcfg.top_k,
        directed=mcfg.directed,
        residual_lambda=float(cfg["graph"].get("residual_lambda", 1.0)),
    )
    return model, fixed


def train_one(mcfg: ModelConfig, fold_id: int, seed: int, cfg: dict, device: torch.device, logger: logging.Logger):
    seed_everything(seed, deterministic=bool(cfg["runtime"].get("deterministic", True)))
    samples = load_validated_samples(cfg, mcfg.lookback)
    idx = split_indices(samples, fold_id)
    if len(idx["train"]) == 0 or len(idx["validation"]) == 0:
        raise Step4DataError(f"Fold {fold_id} has empty train or validation split.")
    source = samples.residual_windows if mcfg.input_kind == "residual" else samples.raw_windows
    scaler = StandardScaler3D().fit(source[idx["train"]])
    x_scaled = scaler.transform(source)
    train_ds = GraphWindowDataset(samples, idx["train"], mcfg.input_kind, x_scaled=x_scaled)
    val_ds = GraphWindowDataset(samples, idx["validation"], mcfg.input_kind, x_scaled=x_scaled)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    model, fixed_adj = build_model_and_adjacency(mcfg, samples, idx["train"], cfg, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["training"].get("scheduler_factor", 0.5)),
        patience=int(cfg["training"].get("scheduler_patience", 5)),
    )
    ckpt_dir = Path(cfg["experiment"]["checkpoint_dir"]) / mcfg.config_id / f"fold_{fold_id}" / f"seed_{seed}"
    hparams = {**asdict(mcfg), "model_id": mcfg.model, "config_id": mcfg.config_id}
    result = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        cfg,
        ckpt_dir,
        hparams,
        seed,
        samples.tickers,
        scaler.state_dict(),
        resume=bool(cfg["runtime"].get("resume", False)),
        logger=logger,
    )
    load_checkpoint(result.checkpoint_path, model, map_location=device)
    raw_pred = predict_loader(
        model,
        val_loader,
        device,
        bool(cfg["runtime"].get("use_amp", False)),
        {**cfg.get("evaluation", {}), "epsilon": cfg.get("target", {}).get("epsilon", 1e-12)},
    )
    pred = flatten_predictions(
        raw_pred,
        samples,
        samples.tickers,
        samples.horizons,
        "validation",
        fold_id,
        seed,
        mcfg.model,
        float(cfg["evaluation"].get("spike_quantile", 0.90)),
    )
    pred["config_id"] = mcfg.config_id
    pred["best_epoch"] = result.best_epoch
    edges = pd.DataFrame()
    adj = model.adjacency()
    if adj is not None:
        edges = adjacency_to_edges(adj, samples.tickers, mcfg.model, fold_id, seed)
        edges["config_id"] = mcfg.config_id
    logger.info("Finished %s fold=%s seed=%s val_qlike=%.6f", mcfg.config_id, fold_id, seed, result.best_metric)
    return pred, edges


def mode_validate_data(cfg: dict, logger: logging.Logger) -> None:
    for lookback in cfg["window"]["lookbacks"]:
        samples = load_validated_samples(cfg, int(lookback))
        logger.info("Validated lookback=%s samples=%s tickers=%s", lookback, len(samples.sample_dates), samples.tickers)


def validation_fold_ids(cfg: dict) -> list[int]:
    samples = load_validated_samples(cfg, int(cfg["window"]["lookbacks"][0]))
    folds = sorted(int(x) for x in np.unique(samples.fold_id[(samples.split == "validation") & (samples.fold_id >= 0)]))
    return folds or [-1]


def mode_train_validation(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    pred_frames, edge_frames = [], []
    for mcfg in enumerate_configs(cfg):
        for fold_id in validation_fold_ids(cfg):
            for seed in cfg["experiment"]["seeds"]:
                if mcfg.model == "G3" and mcfg.graph_seed != seed:
                    continue
                start = time.time()
                try:
                    pred, edges = train_one(mcfg, fold_id, int(seed), cfg, device, logger)
                    pred["runtime_seconds"] = time.time() - start
                    pred_frames.append(pred)
                    if not edges.empty:
                        edge_frames.append(edges)
                except Exception as exc:
                    logger.exception("Failure in %s fold=%s seed=%s", mcfg.config_id, fold_id, seed)
                    failures.append(
                        {
                            "model": mcfg.model,
                            "config_id": mcfg.config_id,
                            "fold_id": fold_id,
                            "seed": seed,
                            "message": str(exc),
                        }
                    )
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    predictions.to_parquet(out_dir / "predictions_validation.parquet", index=False)
    predictions.to_parquet(out_dir / "residual_predictions.parquet", index=False)
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    edges.to_csv(out_dir / "graph_edges.csv", index=False)
    graph_stability(edges).to_csv(out_dir / "graph_stability.csv", index=False)
    pd.DataFrame(failures, columns=["model", "config_id", "fold_id", "seed", "message"]).to_csv(out_dir / "failures.csv", index=False)
    if not predictions.empty:
        write_metric_tables(predictions, out_dir)


def mode_select_config(cfg: dict, logger: logging.Logger) -> dict:
    out_dir = Path(cfg["experiment"]["output_dir"])
    pred_path = out_dir / "predictions_validation.parquet"
    if not pred_path.exists():
        raise Step4DataError("Run train-validation before select-config.")
    pred = pd.read_parquet(pred_path)
    config_scores = pred.groupby(["model", "config_id"], as_index=False)["qlike_loss"].mean().sort_values("qlike_loss")
    best = config_scores.iloc[0].to_dict()
    best_rows = pred.loc[pred["config_id"] == best["config_id"]]
    best_epoch = int(best_rows["best_epoch"].median()) if "best_epoch" in best_rows else int(cfg["training"]["max_epochs"])
    best_doc = {
        "selection_rule": "minimum mean validation QLIKE; locked test not used",
        "model": best["model"],
        "config_id": best["config_id"],
        "validation_qlike": float(best["qlike_loss"]),
        "refit_epochs": max(1, best_epoch + 1),
    }
    with (out_dir / "best_static_graph_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_doc, handle, sort_keys=False)
    logger.info("Selected %s", best_doc)
    return best_doc


def _config_from_id(configs: list[ModelConfig], config_id: str) -> ModelConfig:
    for cfg in configs:
        if cfg.config_id == config_id:
            return cfg
    raise Step4DataError(f"Unknown selected config_id: {config_id}")


def mode_train_final(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    best_path = out_dir / "best_static_graph_config.yaml"
    if not best_path.exists():
        mode_select_config(cfg, logger)
    best = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    mcfg = _config_from_id(enumerate_configs(cfg), best["config_id"])
    final_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    final_cfg["training"]["max_epochs"] = int(best["refit_epochs"])
    for seed in cfg["experiment"]["seeds"]:
        seed_everything(int(seed), bool(cfg["runtime"].get("deterministic", True)))
        samples = load_validated_samples(final_cfg, mcfg.lookback)
        idx = split_indices(samples, None)
        source = samples.residual_windows if mcfg.input_kind == "residual" else samples.raw_windows
        scaler = StandardScaler3D().fit(source[idx["development"]])
        x_scaled = scaler.transform(source)
        dev_ds = GraphWindowDataset(samples, idx["development"], mcfg.input_kind, x_scaled=x_scaled)
        dev_loader = make_loader(dev_ds, final_cfg, shuffle=True)
        model, _ = build_model_and_adjacency(mcfg, samples, idx["development"], final_cfg, int(seed))
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(final_cfg["training"]["learning_rate"]), weight_decay=float(final_cfg["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        ckpt_dir = Path(final_cfg["experiment"]["checkpoint_dir"]) / "final" / mcfg.config_id / f"seed_{seed}"
        train_model(
            model,
            dev_loader,
            dev_loader,
            optimizer,
            scheduler,
            device,
            final_cfg,
            ckpt_dir,
            {**asdict(mcfg), "model_id": mcfg.model, "config_id": mcfg.config_id},
            int(seed),
            samples.tickers,
            scaler.state_dict(),
            resume=bool(final_cfg["runtime"].get("resume", False)),
            logger=logger,
        )


def mode_evaluate_test(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    best = yaml.safe_load((out_dir / "best_static_graph_config.yaml").read_text(encoding="utf-8"))
    mcfg = _config_from_id(enumerate_configs(cfg), best["config_id"])
    frames = []
    for seed in cfg["experiment"]["seeds"]:
        samples = load_validated_samples(cfg, mcfg.lookback)
        idx = split_indices(samples, None)
        source = samples.residual_windows if mcfg.input_kind == "residual" else samples.raw_windows
        scaler = StandardScaler3D().fit(source[idx["development"]])
        x_scaled = scaler.transform(source)
        test_ds = GraphWindowDataset(samples, idx["test"], mcfg.input_kind, x_scaled=x_scaled)
        test_loader = make_loader(test_ds, cfg, shuffle=False)
        model, _ = build_model_and_adjacency(mcfg, samples, idx["development"], cfg, int(seed))
        ckpt = Path(cfg["experiment"]["checkpoint_dir"]) / "final" / mcfg.config_id / f"seed_{seed}" / "best.pt"
        load_checkpoint(ckpt, model, map_location=device)
        model.to(device)
        raw = predict_loader(model, test_loader, device, bool(cfg["runtime"].get("use_amp", False)), {**cfg.get("evaluation", {}), "epsilon": cfg.get("target", {}).get("epsilon", 1e-12)})
        pred = flatten_predictions(raw, samples, samples.tickers, samples.horizons, "test", -1, int(seed), mcfg.model, float(cfg["evaluation"].get("spike_quantile", 0.90)))
        pred["config_id"] = mcfg.config_id
        frames.append(pred)
    test_predictions = pd.concat(frames, ignore_index=True)
    test_predictions.to_parquet(out_dir / "predictions_test.parquet", index=False)
    all_predictions = pd.concat([pd.read_parquet(out_dir / "predictions_validation.parquet"), test_predictions], ignore_index=True)
    all_predictions.to_parquet(out_dir / "residual_predictions.parquet", index=False)
    tables = write_metric_tables(all_predictions, out_dir)
    failures_path = out_dir / "failures.csv"
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    rec_path = out_dir / "masked_reconstruction.csv"
    rec = pd.read_csv(rec_path) if rec_path.exists() else None
    decision, reasons = decide_go_no_go(tables["metrics_by_model"], rec)
    write_report(out_dir, "reports/step4_static_graph_report.md", decision, reasons, failures)
    write_figures(all_predictions, "results")
    logger.info("Test evaluation complete. Decision=%s", decision)


def mode_reconstruction(cfg: dict, device: torch.device, logger: logging.Logger) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    samples = load_validated_samples(cfg, int(cfg["window"]["lookbacks"][0]))
    idx = split_indices(samples, None)
    scaler = StandardScaler3D().fit(samples.residual_windows[idx["development"]])
    x_scaled = scaler.transform(samples.residual_windows)
    top_k = int(cfg["graph"]["top_k"][0])
    corr_adj = correlation_adjacency(samples.residual_windows[idx["development"]], top_k=top_k, directed=False)
    rand_adj = random_adjacency(len(samples.tickers), top_k=top_k, seed=int(cfg["experiment"]["seeds"][0]), directed=False)
    configs = [
        ReconstructionConfig("G1", "identity", torch.eye(len(samples.tickers))),
        ReconstructionConfig("G3", "random", rand_adj, top_k=top_k),
        ReconstructionConfig("G2", "correlation", corr_adj, top_k=top_k),
        ReconstructionConfig("G5", "learned", None, top_k=top_k, graph_embedding_dim=int(cfg["graph"]["embedding_dims"][0])),
    ]
    rec = run_reconstruction_diagnostic(
        x_scaled,
        idx["development"],
        idx["test"] if len(idx["test"]) else idx["development"],
        samples.tickers,
        str(cfg["temporal_encoder"]["options"][0]),
        cfg["temporal_encoder"],
        configs,
        list(cfg["reconstruction"]["mask_ratios"]),
        int(cfg["experiment"]["seeds"][0]),
        device,
        int(cfg["training"]["batch_size"]),
    )
    rec.to_csv(out_dir / "masked_reconstruction.csv", index=False)
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rec.pivot_table(index="model", columns="mask_ratio", values="mse").plot(kind="bar", title="Masked reconstruction MSE")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_masked_reconstruction.png")
    plt.close()
    plot_graph(corr_adj, samples.tickers, fig_dir / "step4_correlation_graph.png", "Correlation graph")
    edge_path = out_dir / "graph_edges.csv"
    if edge_path.exists():
        edges = pd.read_csv(edge_path)
        raw_learned = mean_adjacency_from_edges(edges, samples.tickers, "G4")
        residual_learned = mean_adjacency_from_edges(edges, samples.tickers, "G5")
    else:
        raw_learned = None
        residual_learned = None
    if raw_learned is not None:
        plot_graph(raw_learned, samples.tickers, fig_dir / "step4_learned_raw_graph.png", "Mean learned raw graph G4")
    else:
        plot_unavailable_graph(samples.tickers, fig_dir / "step4_learned_raw_graph.png", "Mean learned raw graph G4")
    if residual_learned is not None:
        plot_graph(residual_learned, samples.tickers, fig_dir / "step4_learned_residual_graph.png", "Mean learned residual graph G5")
    else:
        plot_unavailable_graph(samples.tickers, fig_dir / "step4_learned_residual_graph.png", "Mean learned residual graph G5")
    stability_path = out_dir / "graph_stability.csv"
    if stability_path.exists():
        plot_graph_stability(pd.read_csv(stability_path), fig_dir / "step4_graph_stability.png")
    logger.info("Masked reconstruction diagnostics written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4 static residual graph learning")
    parser.add_argument("--config", default="configs/step4_static_graph.yaml")
    parser.add_argument(
        "--mode",
        choices=["validate-data", "train-validation", "select-config", "train-final", "evaluate-test", "reconstruction", "full"],
        required=True,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = args.num_workers
    if args.resume:
        cfg["runtime"]["resume"] = True
    if args.no_amp:
        cfg["runtime"]["use_amp"] = False
    logger = setup_logger(cfg["experiment"]["log_dir"], args.mode)
    device = resolve_device(str(cfg["runtime"].get("device", "auto")))
    logger.info("Step 4 mode=%s device=%s", args.mode, device)
    if args.mode in {"validate-data", "full"}:
        mode_validate_data(cfg, logger)
    if args.mode in {"train-validation", "full"}:
        mode_train_validation(cfg, device, logger)
    if args.mode in {"select-config", "full"}:
        mode_select_config(cfg, logger)
    if args.mode in {"train-final", "full"}:
        mode_train_final(cfg, device, logger)
    if args.mode in {"reconstruction", "full"} and bool(cfg["reconstruction"].get("enabled", True)):
        mode_reconstruction(cfg, device, logger)
    if args.mode in {"evaluate-test", "full"}:
        mode_evaluate_test(cfg, device, logger)
    summary_path = Path(cfg["experiment"]["output_dir"]) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"mode": args.mode, "device": str(device), "finished": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
