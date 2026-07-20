from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.graph.checkpointing import load_checkpoint, save_checkpoint
from src.graph.losses import residual_loss
from src.graph.metrics import qlike_from_logvol


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    last_epoch: int
    checkpoint_path: Path


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def predict_loader(model, loader: DataLoader, device: torch.device, use_amp: bool, metric_cfg: dict) -> dict[str, np.ndarray]:
    model.eval()
    residual_pred, residual_actual, actual, p_pred, sample_idx = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            with _amp_context(device, use_amp):
                pred = model(x)
            residual_pred.append(pred.detach().cpu().numpy())
            residual_actual.append(batch["y_residual"].numpy())
            actual.append(batch["y_actual"].numpy())
            p_pred.append(batch["p_prediction"].numpy())
            sample_idx.append(batch["sample_index"].numpy())
    residual_pred_arr = np.concatenate(residual_pred, axis=0)
    residual_actual_arr = np.concatenate(residual_actual, axis=0)
    actual_arr = np.concatenate(actual, axis=0)
    p_arr = np.concatenate(p_pred, axis=0)
    final_arr = p_arr + residual_pred_arr
    qlike, clipped = qlike_from_logvol(
        actual_arr,
        final_arr,
        epsilon=float(metric_cfg.get("epsilon", 1e-12)),
        clip_min=float(metric_cfg.get("clip_logvol_min", -20.0)),
        clip_max=float(metric_cfg.get("clip_logvol_max", 20.0)),
    )
    return {
        "sample_index": np.concatenate(sample_idx),
        "residual_prediction": residual_pred_arr,
        "residual_actual": residual_actual_arr,
        "actual_logvol": actual_arr,
        "p_prediction": p_arr,
        "final_prediction": final_arr,
        "qlike_loss": qlike,
        "clipped_predictions": np.asarray([clipped], dtype=np.int64),
    }


def validation_qlike(model, loader: DataLoader, device: torch.device, use_amp: bool, metric_cfg: dict) -> float:
    pred = predict_loader(model, loader, device, use_amp, metric_cfg)
    return float(np.mean(pred["qlike_loss"]))


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: dict,
    checkpoint_dir: str | Path,
    hyperparameters: dict,
    seed: int,
    ticker_order: list[str],
    scaler_state: dict,
    resume: bool = False,
    logger: logging.Logger | None = None,
) -> TrainResult:
    logger = logger or logging.getLogger(__name__)
    training_cfg = cfg["training"]
    metric_cfg = {**cfg.get("evaluation", {}), "epsilon": cfg.get("target", {}).get("epsilon", 1e-12)}
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    max_epochs = int(training_cfg["max_epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    grad_clip = float(training_cfg.get("gradient_clip_norm", 0.0))
    loss_name = str(hyperparameters["loss"])
    use_amp = bool(cfg["runtime"].get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    if resume and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_validation_metric"])
        best_epoch = int(checkpoint.get("best_epoch", -1))
        logger.info("Resumed checkpoint %s at epoch %s", last_path, start_epoch)

    epochs_without_improvement = 0
    model.to(device)
    for epoch in tqdm(range(start_epoch, max_epochs), desc=f"train {hyperparameters['model_id']} seed={seed}", leave=False):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            y = batch["y_residual"].to(device)
            with _amp_context(device, use_amp):
                pred = model(x)
                loss = residual_loss(pred, y, loss_name)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        val_metric = validation_qlike(model, val_loader, device, use_amp, metric_cfg)
        if scheduler is not None:
            scheduler.step(val_metric)
        adjacency = model.adjacency() if hasattr(model, "adjacency") else None
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            min(best_metric, val_metric),
            hyperparameters,
            seed,
            ticker_order,
            scaler_state,
            adjacency,
        )
        if val_metric < best_metric:
            best_metric = val_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                hyperparameters,
                seed,
                ticker_order,
                scaler_state,
                adjacency,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return TrainResult(best_metric=best_metric, best_epoch=best_epoch, last_epoch=epoch, checkpoint_path=best_path)

