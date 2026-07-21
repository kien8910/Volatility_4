from __future__ import annotations

import torch
from torch import nn


class EMAGraphBuffer(nn.Module):
    """Non-gradient EMA buffer for dense graph scores before Top-k."""

    def __init__(self, shape: tuple[int, ...], beta: float):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError("EMA beta must be in [0, 1].")
        self.beta = float(beta)
        self.register_buffer("ema_score", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update(self, current_score: torch.Tensor) -> torch.Tensor:
        current = current_score.detach()
        if self.beta == 0.0 or not bool(self.initialized.item()):
            self.ema_score.copy_(current)
        else:
            self.ema_score.mul_(self.beta).add_(current, alpha=1.0 - self.beta)
        self.initialized.fill_(True)
        return self.ema_score

    def value_or_current(self, current_score: torch.Tensor) -> torch.Tensor:
        if self.beta == 0.0 or not bool(self.initialized.item()):
            return current_score
        return self.ema_score.to(device=current_score.device, dtype=current_score.dtype)

