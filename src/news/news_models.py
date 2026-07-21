from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.models import StaticGraphForecastModel
from src.news.additive_fusion import HierarchicalAdditiveFusion
from src.news.concatenation_fusion import EarlyConcatenationFusion
from src.news.heterogeneous_fusion import HeterogeneousRelationFusion
from src.news.hierarchy_projection import HierarchyProjection


class ExposedStaticBackbone(nn.Module):
    def __init__(self, backbone: StaticGraphForecastModel):
        super().__init__()
        self.backbone = backbone

    @property
    def hidden_dim(self) -> int:
        first = self.backbone.head[0]
        return int(first.in_features)

    def adjacency(self) -> torch.Tensor | None:
        return self.backbone.adjacency()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.temporal(x)
        return self.backbone.gcn(h, self.backbone.adjacency())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.forward_features(x)
        residual = self.backbone.head(h)
        return {
            "stock_node_embedding": h,
            "stock_residual_prediction": residual,
            "static_adjacency": self.backbone.adjacency(),
        }


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


@dataclass(frozen=True)
class Step6ModelConfig:
    model: str
    ablation: str
    pooling_method: str
    projection_dim: int

    @property
    def config_id(self) -> str:
        return f"N__{self.model}__{self.ablation}__pool{self.pooling_method}__dp{self.projection_dim}"

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "ablation": self.ablation,
            "pooling_method": self.pooling_method,
            "projection_dim": self.projection_dim,
            "config_id": self.config_id,
        }


class NaiveNewsFusionModel(nn.Module):
    def __init__(
        self,
        stock_backbone: ExposedStaticBackbone,
        model: str,
        hierarchies: list[str],
        embedding_dim: int,
        projection_dim: int,
        control_dim: int,
        hidden_dims: list[int],
        num_horizons: int,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if model not in {"stock_only", "concatenation", "hierarchical_additive", "heterogeneous"}:
            raise ValueError(f"Unsupported Step 6 model: {model}")
        self.model = model
        self.hierarchies = list(hierarchies)
        self.stock_backbone = stock_backbone
        if freeze_backbone:
            freeze_module(self.stock_backbone)
        self.projection = HierarchyProjection(hierarchies, embedding_dim, projection_dim, dropout)
        stock_dim = stock_backbone.hidden_dim
        if model == "concatenation":
            self.fusion = EarlyConcatenationFusion(stock_dim, projection_dim, control_dim, len(hierarchies), hidden_dims, num_horizons, dropout)
        elif model == "hierarchical_additive":
            self.fusion = HierarchicalAdditiveFusion(stock_dim, projection_dim, hidden_dims, num_horizons, dropout)
        elif model == "heterogeneous":
            self.fusion = HeterogeneousRelationFusion(stock_dim, projection_dim, hidden_dims, num_horizons, dropout)
        else:
            self.fusion = None

    def forward(self, x: torch.Tensor, news_embeddings: torch.Tensor, news_controls: torch.Tensor) -> dict[str, torch.Tensor]:
        stock = self.stock_backbone(x)
        stock_pred = stock["stock_residual_prediction"]
        if self.model == "stock_only":
            correction = torch.zeros_like(stock_pred)
        else:
            messages = self.projection(news_embeddings)
            if self.model == "concatenation":
                correction = self.fusion(stock["stock_node_embedding"], messages, news_controls)
            elif self.model == "hierarchical_additive":
                residual = self.fusion(stock["stock_node_embedding"], messages)
                correction = residual - stock_pred
            else:
                residual = self.fusion(stock["stock_node_embedding"], messages, stock["static_adjacency"])
                correction = residual - stock_pred
        final_residual = stock_pred + correction
        return {
            "stock_node_embedding": stock["stock_node_embedding"],
            "static_adjacency": stock["static_adjacency"],
            "stock_residual_prediction": stock_pred,
            "news_residual_correction": correction,
            "final_residual_prediction": final_residual,
        }

