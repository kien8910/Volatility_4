from __future__ import annotations
import torch
import torch.nn.functional as F


def qlike_logvol(actual: torch.Tensor, predicted: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    ratio = torch.exp(2.0 * (actual.clamp(-clip, clip) - predicted.clamp(-clip, clip)))
    return (ratio - torch.log(ratio) - 1.0).mean()


def sparse_hurdle_loss(actual: torch.Tensor, prediction: torch.Tensor, correction: torch.Tensor,
                       hurdle_probability: torch.Tensor, impact_label: torch.Tensor,
                       edge_gate: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    main = qlike_logvol(actual, prediction)
    hurdle = (F.binary_cross_entropy(hurdle_probability.clamp(1e-6, 1 - 1e-6), impact_label)
              if hurdle_probability.numel() else correction.sum() * 0.0)
    correction_penalty = correction.square().mean()
    gate_penalty = edge_gate.mean() if edge_gate.numel() else correction.sum() * 0.0
    total = (main + float(cfg["regularization"]["hurdle_bce_weight"]) * hurdle
             + float(cfg["regularization"]["correction_l2_weight"]) * correction_penalty
             + float(cfg["regularization"]["gate_mean_weight"]) * gate_penalty)
    return total, {"main_qlike": main, "hurdle_bce": hurdle,
                   "correction_l2": correction_penalty, "gate_mean": gate_penalty}
