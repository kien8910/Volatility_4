import torch

from src.news.news_models import NaiveNewsFusionModel


class DummyBackbone(torch.nn.Module):
    hidden_dim = 4

    def adjacency(self):
        return torch.eye(3)

    def forward(self, x):
        batch = x.shape[0]
        h = torch.ones(batch, 3, 4)
        return {"stock_node_embedding": h, "stock_residual_prediction": torch.zeros(batch, 3, 2), "static_adjacency": self.adjacency()}


def test_step6_fusion_output_shapes():
    for model_name in ["stock_only", "concatenation", "hierarchical_additive", "heterogeneous"]:
        model = NaiveNewsFusionModel(DummyBackbone(), model_name, ["macro", "sector"], 5, 4, 3, [6], 2)
        out = model(torch.zeros(7, 3, 22), torch.randn(7, 3, 2, 5), torch.randn(7, 3, 3))
        assert out["final_residual_prediction"].shape == (7, 3, 2)
        assert out["news_residual_correction"].shape == (7, 3, 2)

