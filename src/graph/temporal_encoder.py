from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class TemporalLinear(nn.Module):
    def __init__(self, lookback: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, nodes, lookback], got {tuple(x.shape)}")
        return self.net(x)


class SmallTCN(nn.Module):
    def __init__(
        self,
        lookback: int,
        hidden_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            padding = kernel_size - 1
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    _activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(channels[-1], hidden_dim), nn.LayerNorm(hidden_dim))
        self.lookback = lookback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, nodes, lookback], got {tuple(x.shape)}")
        b, n, l = x.shape
        z = x.reshape(b * n, 1, l)
        z = self.conv(z)
        z = z[:, :, :l]
        z = z[:, :, -1]
        return self.proj(z).reshape(b, n, -1)


def build_temporal_encoder(kind: str, lookback: int, cfg: dict) -> nn.Module:
    hidden_dim = int(cfg["hidden_dim"])
    if kind == "linear":
        return TemporalLinear(lookback=lookback, hidden_dim=hidden_dim)
    if kind == "small_tcn":
        return SmallTCN(
            lookback=lookback,
            hidden_dim=hidden_dim,
            channels=list(cfg.get("channels", [32, 32])),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "relu")),
        )
    raise ValueError(f"Unknown temporal encoder: {kind}")

