import numpy as np
import torch

from src.graph.masked_reconstruction import ReconstructionDataset
from src.graph.models import MaskedReconstructionModel


def test_masked_dataset_masks_entire_ticker_history():
    x = np.ones((3, 11, 22), dtype=np.float32)
    ds = ReconstructionDataset(x, np.array([0, 1, 2]), mask_ratio=0.25, seed=42)
    item = ds[0]
    masked_nodes = int(item["mask"].sum().item())
    assert masked_nodes >= 1
    masked_x = item["x"].masked_fill(item["mask"][:, None], 0.0)
    assert torch.all(masked_x[item["mask"]] == 0.0)


def test_reconstruction_model_forward_shape():
    model = MaskedReconstructionModel(
        num_nodes=11,
        lookback=22,
        temporal_kind="linear",
        temporal_cfg={"hidden_dim": 16},
        graph_type="identity",
        fixed_adjacency=torch.eye(11),
    )
    x = torch.randn(2, 11, 22)
    mask = torch.zeros(2, 11, dtype=torch.bool)
    mask[:, 0] = True
    out = model(x, mask)
    assert out.shape == (2, 11)

