from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    hyperparameters: dict,
    seed: int,
    ticker_order: list[str],
    scaler_state: dict,
    adjacency: torch.Tensor | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "best_validation_metric": float(best_metric),
            "hyperparameters": hyperparameters,
            "random_seed": int(seed),
            "ticker_order": ticker_order,
            "scaler_parameters": scaler_state,
            "adjacency": adjacency.detach().cpu() if adjacency is not None else None,
            "git_commit_hash": git_commit_hash(),
        },
        path,
    )


def _resolve_torch_device(map_location) -> torch.device | None:
    if map_location is None:
        return None
    if isinstance(map_location, torch.device):
        return map_location
    if isinstance(map_location, str):
        return torch.device(map_location)
    return None


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    device = _resolve_torch_device(map_location)
    if device is not None:
        model.to(device)
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model.to(device)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if device is not None:
            _move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
