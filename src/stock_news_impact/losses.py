from __future__ import annotations

import torch


def gated_forecast_loss(final_prediction: torch.Tensor, actual_logvol: torch.Tensor) -> torch.Tensor:
    return torch.mean((final_prediction - actual_logvol) ** 2)


def gate_regularization(final_gate: torch.Tensor, gated_correction: torch.Tensor, usage_weight: float, correction_weight: float) -> torch.Tensor:
    return float(usage_weight) * final_gate.mean() + float(correction_weight) * torch.mean(gated_correction**2)
