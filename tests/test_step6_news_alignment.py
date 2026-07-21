import numpy as np
import pandas as pd

from src.news.news_dataset import align_news_to_samples


class DummySamples:
    sample_dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    tickers = ["ADI", "AMD"]
    horizons = [1]


def _features():
    rows = []
    for date in DummySamples.sample_dates:
        for ticker in DummySamples.tickers:
            row = {"date": date, "ticker": ticker, "has_any_dynamic_news": 1}
            for h in ["macro", "sector", "target_company", "related_company", "filing"]:
                row.update(
                    {
                        f"{h}_text_hash": f"{h}-{date.date()}",
                        f"{h}_missing_mask": 0,
                        f"has_{h}": 1,
                        f"{h}_count": 1,
                        f"{h}_text_char_length": 10,
                        f"{h}_text_word_count": 2,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_alignment_uses_forecast_origin_date_only():
    emb = pd.DataFrame(
        [
            {
                "text_hash": f"{h}-{date.date()}",
                "hierarchy": h,
                "encoder_name": "hashing-test",
                "pooling_method": "cls",
                "max_length": 256,
                "embedding_dim": 2,
                "embedding": [float(i), 0.0],
                "missing_mask": 0,
                "created_at": "now",
            }
            for i, date in enumerate(DummySamples.sample_dates)
            for h in ["macro", "sector", "target_company", "related_company", "filing"]
        ]
    )
    aligned = align_news_to_samples(DummySamples, _features(), emb, "hashing-test", "cls", 256)
    assert np.allclose(aligned.embeddings[0, :, 0, 0], 0.0)
    assert np.allclose(aligned.embeddings[1, :, 0, 0], 1.0)

