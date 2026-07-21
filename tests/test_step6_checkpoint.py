import torch

from src.news.checkpointing import load_step6_checkpoint, save_step6_checkpoint


def test_step6_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "best.pt"
    save_step6_checkpoint(path, model, opt, None, 2, 0.5, {"config_id": "c"}, 42, ["ADI"], {"mean": [[[0.0]]], "scale": [[[1.0]]]})
    loaded = torch.nn.Linear(2, 1)
    ckpt = load_step6_checkpoint(path, loaded)
    assert ckpt["epoch"] == 2
    for lhs, rhs in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(lhs, rhs)

