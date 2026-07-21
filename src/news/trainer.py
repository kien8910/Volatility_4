from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.graph.losses import residual_loss
from src.graph.metrics import qlike_from_logvol
from src.news.checkpointing import load_step6_checkpoint, save_step6_checkpoint


@dataclass
class Step6TrainResult:
    best_metric: float
    best_epoch: int
    last_epoch: int
    checkpoint_path: Path


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def make_grad_scaler(device: torch.device, enabled: bool):
    use = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use)


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(trainable), int(total)


def predict_step6_loader(model, loader: DataLoader, device: torch.device, use_amp: bool, metric_cfg: dict) -> dict[str, np.ndarray]:
    model.eval()
    sample_idx, stock_pred, correction, final_res, residual_actual, actual, p_pred = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            emb = batch["news_embeddings"].to(device)
            controls = batch["news_controls"].to(device)
            with amp_context(device, use_amp):
                out = model(x, emb, controls)
            sample_idx.append(batch["sample_index"].numpy())
            stock_pred.append(out["stock_residual_prediction"].detach().cpu().numpy())
            correction.append(out["news_residual_correction"].detach().cpu().numpy())
            final_res.append(out["final_residual_prediction"].detach().cpu().numpy())
            residual_actual.append(batch["y_residual"].numpy())
            actual.append(batch["y_actual"].numpy())
            p_pred.append(batch["p_prediction"].numpy())
    stock_arr = np.concatenate(stock_pred, axis=0)
    correction_arr = np.concatenate(correction, axis=0)
    final_res_arr = np.concatenate(final_res, axis=0)
    actual_arr = np.concatenate(actual, axis=0)
    p_arr = np.concatenate(p_pred, axis=0)
    final_arr = p_arr + final_res_arr
    qlike, clipped = qlike_from_logvol(
        actual_arr,
        final_arr,
        epsilon=float(metric_cfg.get("epsilon", 1e-12)),
        clip_min=float(metric_cfg.get("clip_logvol_min", -20.0)),
        clip_max=float(metric_cfg.get("clip_logvol_max", 20.0)),
    )
    return {
        "sample_index": np.concatenate(sample_idx),
        "stock_residual_prediction": stock_arr,
        "news_residual_correction": correction_arr,
        "final_residual_prediction": final_res_arr,
        "residual_actual": np.concatenate(residual_actual, axis=0),
        "actual_logvol": actual_arr,
        "p_prediction": p_arr,
        "final_prediction": final_arr,
        "qlike_loss": qlike,
        "clipped_predictions": np.asarray([clipped], dtype=np.int64),
    }


def validation_qlike(model, loader: DataLoader, device: torch.device, use_amp: bool, metric_cfg: dict) -> float:
    raw = predict_step6_loader(model, loader, device, use_amp, metric_cfg)
    return float(np.mean(raw["qlike_loss"]))


def train_step6_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: dict,
    checkpoint_dir: str | Path,
    run_config: dict,
    seed: int,
    ticker_order: list[str],
    scaler_state: dict,
    resume: bool = False,
    logger: logging.Logger | None = None,
) -> Step6TrainResult:
    logger = logger or logging.getLogger(__name__)
    training_cfg = cfg["training"]
    metric_cfg = {**cfg.get("evaluation", {}), "epsilon": cfg.get("target", {}).get("epsilon", 1e-12)}
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    max_epochs = int(training_cfg["max_epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    grad_clip = float(training_cfg.get("gradient_clip_norm", 0.0))
    use_amp = bool(cfg["runtime"].get("use_amp", False))
    scaler = make_grad_scaler(device, use_amp)
    model.to(device)
    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    if resume and last_path.exists():
        checkpoint_meta = torch.load(last_path, map_location="cpu", weights_only=False)
        checkpoint_config_id = str(checkpoint_meta.get("run_config", {}).get("config_id", ""))
        current_config_id = str(run_config.get("config_id", ""))
        if checkpoint_config_id and checkpoint_config_id != current_config_id:
            logger.warning(
                "Ignoring incompatible Step 6 checkpoint %s: checkpoint config_id=%s current config_id=%s",
                last_path,
                checkpoint_config_id,
                current_config_id,
            )
        else:
            checkpoint = load_step6_checkpoint(last_path, model, optimizer, scheduler, map_location=device)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint["best_validation_metric"])
            best_epoch = int(checkpoint.get("best_epoch", -1))
            logger.info("Resumed Step 6 checkpoint %s at epoch %s", last_path, start_epoch)
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    for epoch in tqdm(range(start_epoch, max_epochs), desc=f"train {run_config['config_id']} seed={seed}", leave=False):
        last_epoch = epoch
        model.train()
        model.stock_backbone.eval()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            emb = batch["news_embeddings"].to(device)
            controls = batch["news_controls"].to(device)
            y = batch["y_residual"].to(device)
            with amp_context(device, use_amp):
                out = model(x, emb, controls)
                loss = residual_loss(out["final_residual_prediction"], y, str(training_cfg.get("loss", "mse")))
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
        val_metric = validation_qlike(model, val_loader, device, use_amp, metric_cfg)
        if scheduler is not None:
            scheduler.step(val_metric)
        save_step6_checkpoint(last_path, model, optimizer, scheduler, epoch, min(best_metric, val_metric), run_config, seed, ticker_order, scaler_state)
        if val_metric < best_metric:
            best_metric = val_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_step6_checkpoint(best_path, model, optimizer, scheduler, epoch, best_metric, run_config, seed, ticker_order, scaler_state)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return Step6TrainResult(best_metric, best_epoch, last_epoch, best_path)
