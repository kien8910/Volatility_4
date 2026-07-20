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


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint

