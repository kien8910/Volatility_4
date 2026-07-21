import numpy as np

from src.news.text_encoder import HashingTextEncoder


def test_hashing_encoder_reproducible_for_same_seedless_input():
    enc = HashingTextEncoder(embedding_dim=8)
    a = enc.encode(["NVDA earnings"], "cls")
    b = enc.encode(["NVDA earnings"], "cls")
    assert np.allclose(a, b)

