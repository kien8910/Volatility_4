from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.graph.losses import residual_loss
from src.graph.metrics import qlike_from_logvol
from src.regime_graph.checkpointing import load_step5_checkpoint, save_step5_checkpoint
from src.regime_graph.regularization import gate_regularization


class Step5WindowDataset(Dataset):
    def __init__(self, base_dataset, state_features: np.ndarray, market_states: np.ndarray):
        self.base_dataset = base_dataset
        self.state_features = state_features.astype(np.float32)
        self.market_states = market_states.astype(object)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        sample_index = int(item["sample_index"].item())
        item["state_features"] = torch.from_numpy(self.state_features[sample_index]).float()
        item["market_state_code"] = torch.tensor({"low_volatility": 0, "medium_volatility": 1, "high_volatility": 2}[self.market_states[sample_index]], dtype=torch.long)
        return item


@dataclass
class Step5TrainResult:
    best_metric: float
    best_epoch: int
    last_epoch: int
    checkpoint_path: Path


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def make_grad_scaler(device: torch.device, enabled: bool):
    active = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


def predict_step5_loader(model, loader: DataLoader, device: torch.device, use_amp: bool, metric_cfg: dict) -> dict[str, np.ndarray]:
    model.eval()
    residual_pred, residual_actual, actual, p_pred, sample_idx = [], [], [], [], []
    weights, states = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            sf = batch["state_features"].to(device)
            with amp_context(device, use_amp):
                pred, aux = model(x, sf)
            residual_pred.append(pred.detach().cpu().numpy())
            residual_actual.append(batch["y_residual"].numpy())
            actual.append(batch["y_actual"].numpy())
            p_pred.append(batch["p_prediction"].numpy())
            sample_idx.append(batch["sample_index"].numpy())
            weights.append(aux["regime_weights"].detach().cpu().numpy())
            states.append(batch["market_state_code"].numpy())
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
        "regime_weights": np.concatenate(weights, axis=0),
        "market_state_code": np.concatenate(states, axis=0),
        "clipped_predictions": np.asarray([clipped], dtype=np.int64),
    }


def validation_qlike(model, loader: DataLoader, device: torch.device, use_amp: bool, metric_cfg: dict) -> float:
    raw = predict_step5_loader(model, loader, device, use_amp, metric_cfg)
    return float(np.mean(raw["qlike_loss"]))


def train_step5_model(
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
    input_scaler_state: dict,
    state_scaler_state: dict,
    resume: bool = False,
    logger: logging.Logger | None = None,
) -> Step5TrainResult:
    logger = logger or logging.getLogger(__name__)
    training_cfg = cfg["training"]
    metric_cfg = cfg["evaluation"]
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    use_amp = bool(cfg["runtime"].get("use_amp", False))
    scaler = make_grad_scaler(device, use_amp)
    model.to(device)
    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    if resume and last_path.exists():
        checkpoint = load_step5_checkpoint(last_path, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_validation_qlike"])
        best_epoch = int(checkpoint.get("best_epoch", -1))
        logger.info("Resumed Step 5 checkpoint %s at epoch %s", last_path, start_epoch)

    epochs_without_improvement = 0
    max_epochs = int(training_cfg["max_epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    grad_clip = float(training_cfg.get("gradient_clip_norm", 0.0))
    loss_name = str(run_config.get("loss", training_cfg.get("loss", "mse")))
    reg_option = str(run_config.get("gate_regularization", "none"))
    reg_weight = float(run_config.get("gate_regularization_weight", 0.0))
    for epoch in tqdm(range(start_epoch, max_epochs), desc=f"train {run_config['config_id']} seed={seed}", leave=False):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            sf = batch["state_features"].to(device)
            y = batch["y_residual"].to(device)
            with amp_context(device, use_amp):
                pred, aux = model(x, sf)
                loss = residual_loss(pred, y, loss_name)
                loss = loss + gate_regularization(aux["regime_weights"], reg_option, reg_weight)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        model.update_ema()
        val_metric = validation_qlike(model, val_loader, device, use_amp, metric_cfg)
        if scheduler is not None:
            scheduler.step(val_metric)
        save_step5_checkpoint(last_path, model, optimizer, scheduler, epoch, min(best_metric, val_metric), run_config, seed, ticker_order, input_scaler_state, state_scaler_state)
        if val_metric < best_metric:
            best_metric = val_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_step5_checkpoint(best_path, model, optimizer, scheduler, epoch, best_metric, run_config, seed, ticker_order, input_scaler_state, state_scaler_state)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return Step5TrainResult(best_metric=best_metric, best_epoch=best_epoch, last_epoch=epoch, checkpoint_path=best_path)
