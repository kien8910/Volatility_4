import numpy as np
import pandas as pd

from src.news.embedding_cache import EmbeddingCache, embedding_lookup
from src.news.text_encoder import HashingTextEncoder


def test_embedding_cache_deduplicates_and_zeroes_missing(tmp_path):
    cache = EmbeddingCache(tmp_path)
    encoder = HashingTextEncoder(embedding_dim=8)
    requests = pd.DataFrame(
        [
            {"hierarchy": "macro", "text_hash": "h1", "text": "same text", "missing_mask": 0},
            {"hierarchy": "macro", "text_hash": "h1", "text": "same text", "missing_mask": 0},
            {"hierarchy": "sector", "text_hash": "empty", "text": "", "missing_mask": 1},
        ]
    )
    frame = cache.get_or_encode(requests, encoder, "hashing-test", "cls", 256)
    assert len(frame) == 2
    lookup = embedding_lookup(frame, "hashing-test", "cls", 256)
    assert np.allclose(lookup["empty"], np.zeros(8))
    frame2 = cache.get_or_encode(requests, encoder, "hashing-test", "cls", 256)
    assert len(frame2) == 2


def test_embedding_cache_separates_pooling_methods(tmp_path):
    cache = EmbeddingCache(tmp_path)
    encoder = HashingTextEncoder(embedding_dim=8)
    req = pd.DataFrame([{"hierarchy": "macro", "text_hash": "h1", "text": "text", "missing_mask": 0}])
    cache.get_or_encode(req, encoder, "hashing-test", "cls", 256)
    frame = cache.get_or_encode(req, encoder, "hashing-test", "mean", 256)
    assert set(frame["pooling_method"]) == {"cls", "mean"}

