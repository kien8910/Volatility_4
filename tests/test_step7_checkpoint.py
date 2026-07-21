import torch

from src.stock_news_impact.checkpointing import load_step7_checkpoint, save_step7_checkpoint


def test_step7_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "best.pt"
    save_step7_checkpoint(path, model, opt, 1, 0.2, {"config_id": "c"})
    loaded = torch.nn.Linear(2, 1)
    ckpt = load_step7_checkpoint(path, loaded)
    assert ckpt["epoch"] == 1
    for a, b in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(a, b)

