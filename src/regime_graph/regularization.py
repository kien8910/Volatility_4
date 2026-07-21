from __future__ import annotations

import torch


def gate_regularization(weights: torch.Tensor, option: str, strength: float, epsilon: float = 1e-12) -> torch.Tensor:
    if option == "none" or strength == 0:
        return weights.new_tensor(0.0)
    mean_usage = weights.mean(dim=0)
    if option == "entropy":
        entropy = -(mean_usage * torch.log(mean_usage + epsilon)).sum()
        return -float(strength) * entropy
    if option == "load_balance":
        target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        return float(strength) * torch.mean((mean_usage - target) ** 2)
    raise ValueError(f"Unsupported gate regularization: {option}")

