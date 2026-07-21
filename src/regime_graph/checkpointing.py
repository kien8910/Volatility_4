from __future__ import annotations

from pathlib import Path

import torch

from src.graph.checkpointing import git_commit_hash


def save_step5_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config: dict,
    seed: int,
    ticker_order: list[str],
    input_scaler_state: dict,
    state_scaler_state: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "best_validation_qlike": float(best_metric),
            "config": config,
            "seed": int(seed),
            "ticker_order": ticker_order,
            "input_scaler": input_scaler_state,
            "state_scaler": state_scaler_state,
            "current_graph_scores": model.current_dense_scores().detach().cpu() if hasattr(model, "current_dense_scores") else None,
            "ema_graph_scores": model.ema.ema_score.detach().cpu() if hasattr(model, "ema") else None,
            "git_commit_hash": git_commit_hash(),
        },
        path,
    )


def load_step5_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    device = torch.device(map_location) if isinstance(map_location, str) else map_location
    if isinstance(device, torch.device):
        model.to(device)
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if isinstance(device, torch.device):
        model.to(device)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if isinstance(device, torch.device):
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint

