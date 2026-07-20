from __future__ import annotations

import torch
import torch.nn.functional as F


def residual_loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "huber":
        return F.huber_loss(prediction, target, delta=1.0)
    raise ValueError(f"Unsupported loss: {loss_name}")

