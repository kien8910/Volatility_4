import torch

from src.news.news_models import NaiveNewsFusionModel


class ParamBackbone(torch.nn.Module):
    hidden_dim = 4

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        batch = x.shape[0]
        return {"stock_node_embedding": torch.ones(batch, 2, 4), "stock_residual_prediction": torch.zeros(batch, 2, 1), "static_adjacency": None}


def test_stock_backbone_frozen_in_main_model():
    backbone = ParamBackbone()
    NaiveNewsFusionModel(backbone, "concatenation", ["macro"], 3, 4, 2, [5], 1, freeze_backbone=True)
    assert all(not p.requires_grad for p in backbone.parameters())

