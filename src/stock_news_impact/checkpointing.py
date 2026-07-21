from __future__ import annotations

from pathlib import Path

import torch


def save_step7_checkpoint(path: str | Path, model: torch.nn.Module, optimizer, epoch: int, metric: float, run_config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best_validation_metric": float(metric),
            "run_config": run_config,
        },
        path,
    )


def load_step7_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
