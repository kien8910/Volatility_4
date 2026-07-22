from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.news.embedding_cache import EmbeddingCache
from src.sparse_target_text.chunking import tokenizer_chunk_events
from src.sparse_target_text.data import (attach_embeddings_and_novelty, _effective_dates,
                                         shuffle_event_payload_within_day)
from src.sparse_target_text.filtering import hard_filter_events
from src.sparse_target_text.losses import sparse_hurdle_loss
from src.sparse_target_text.models import SparseHurdleCorrector, segment_topk_mask
from src.sparse_target_text.trainer import STATE_COLUMNS, _tensor_bundle, outputs_to_frames


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False): return 2
    def encode(self, text, add_special_tokens=False, truncation=False): return list(range(len(text.split())))
    def decode(self, ids, **kwargs): return " ".join(f"t{x}" for x in ids)


def minimal_cfg():
    return {
        "information_cutoff": {"news_lag_sessions": 1},
        "event_filter": {"min_words": 1, "max_words": 100, "require_catalyst_keyword": False,
                         "catalyst_keywords": {"earnings": ["earnings"]}},
        "text_encoder": {"max_length": 6},
        "model": {"hidden_dim": 8, "stock_embedding_dim": 3, "horizon_embedding_dim": 2,
                  "event_type_embedding_dim": 2, "event_type_count": 8, "alpha_init": .005,
                  "alpha_max": .05, "edge_gate_init_probability": .1, "hurdle_init_probability": .05},
        "regularization": {"hurdle_bce_weight": .02, "correction_l2_weight": .001, "gate_mean_weight": .001},
    }


def test_missing_timestamp_news_moves_to_next_forecast_session():
    baseline = pd.DataFrame({"ticker": ["AMD"] * 3, "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])})
    events = pd.DataFrame({"ticker": ["AMD"], "news_date": pd.to_datetime(["2022-01-03"])})
    assert _effective_dates(events, baseline, 1).iloc[0] == pd.Timestamp("2022-01-04")


def test_tokenizer_chunks_never_exceed_payload():
    event = pd.DataFrame({"event_id": ["e"], "text_hash": ["h"], "text": ["one two three four five six seven"]})
    chunks = tokenizer_chunk_events(event, minimal_cfg(), FakeTokenizer())
    assert chunks.token_count.max() <= 4
    assert chunks.chunk_count.iloc[0] == 2


def test_topk_selects_at_most_k_per_row():
    scores = torch.tensor([.1, .9, .2, .8, .7]); rows = torch.tensor([0, 0, 0, 1, 1])
    mask = segment_topk_mask(scores, rows, 1)
    assert mask.tolist() == [0, 1, 0, 1, 0]


def test_no_event_rows_have_exact_zero_correction():
    cfg = minimal_cfg(); net = SparseHurdleCorrector(4, 3, 2, (1, 5), cfg)
    output = net(torch.empty(0, 4), torch.empty(0, 5), torch.empty(0, dtype=torch.long),
                 torch.empty(0, dtype=torch.long), torch.zeros(2, 3), torch.tensor([0, 1]),
                 torch.tensor([1, 5]), "learned_topk", 1)
    assert torch.equal(output["correction"], torch.zeros(2))
    assert torch.equal(output["hurdle_probability"], torch.zeros(2))


def test_correction_is_bounded_by_alpha_max():
    cfg = minimal_cfg(); net = SparseHurdleCorrector(4, 3, 1, (1,), cfg)
    output = net(torch.randn(2, 4), torch.ones(2, 5), torch.zeros(2, dtype=torch.long),
                 torch.zeros(2, dtype=torch.long), torch.zeros(1, 3), torch.zeros(1, dtype=torch.long),
                 torch.ones(1, dtype=torch.long), "all", 1)
    assert output["correction"].abs().max() <= cfg["model"]["alpha_max"] + 1e-7


def test_empty_hurdle_batch_loss_is_finite():
    total, parts = sparse_hurdle_loss(torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]),
                                      torch.empty(0), torch.empty(0), torch.empty(0), minimal_cfg())
    assert torch.isfinite(total)
    assert all(torch.isfinite(x) for x in parts.values())


def test_placebo_is_deterministic_and_tracks_wrong_ticker_payload():
    events = pd.DataFrame({"effective_date": pd.to_datetime(["2022-01-03"] * 3),
        "ticker": ["AMD", "NVDA", "INTC"], "text_hash": ["a", "b", "c"],
        "embedding": [np.ones(2), np.ones(2)*2, np.ones(2)*3], "event_type": ["x"]*3,
        "catalyst_score": [1.]*3, "semantic_novelty": [.1, .2, .3], "word_count": [10]*3})
    a, da = shuffle_event_payload_within_day(events, 42); b, db = shuffle_event_payload_within_day(events, 42)
    assert a.text_hash.tolist() == b.text_hash.tolist()
    assert da == db
    assert a.wrong_ticker_payload.mean() == 1.0


def test_hard_filter_preserves_direct_target_semantics():
    event = pd.DataFrame({"event_id": ["e"], "news_date": pd.to_datetime(["2022-01-01"]),
        "effective_date": pd.to_datetime(["2022-01-03"]), "ticker": ["AMD"], "text": ["earnings rose"],
        "text_hash": ["h"], "category": ["targetCompany_category1"], "publication_timestamp": [pd.NaT]})
    out = hard_filter_events(event, minimal_cfg())
    assert out.entity_relevance.iloc[0] == 1.0
    assert out.timestamp_confidence.iloc[0] == 0.0


def test_embedding_attach_ignores_events_rejected_by_basic_filter(tmp_path, monkeypatch):
    cfg = minimal_cfg()
    cfg["text_encoder"].update({
        "cache_dir": str(tmp_path), "model_name": "test-encoder", "pooling_method": "cls"
    })
    cache = pd.DataFrame({
        "text_hash": ["eligible"], "hierarchy": ["target_company"],
        "encoder_name": ["test-encoder"], "pooling_method": ["cls"], "max_length": [6],
        "embedding_dim": [2], "embedding": [[1.0, 0.0]], "missing_mask": [0],
        "created_at": [pd.Timestamp.utcnow().isoformat()],
    })
    EmbeddingCache(tmp_path).write(cache)
    events = pd.DataFrame({
        "event_id": ["e1", "e2"], "effective_date": pd.to_datetime(["2022-01-03"] * 2),
        "ticker": ["AMD", "AMD"], "text": ["eligible text", "short"],
        "text_hash": ["eligible", "rejected"], "basic_filter_pass": [True, False],
    })
    monkeypatch.setattr(
        "src.sparse_target_text.data.tokenizer_chunk_events", lambda frame, _cfg: frame.copy()
    )

    attached = attach_embeddings_and_novelty(events, cfg)

    assert attached.event_id.tolist() == ["e1"]
    assert np.allclose(attached.embedding.iloc[0], [1.0, 0.0])


def test_tensor_bundle_keeps_news_row_id_separate_from_prediction_row_id():
    cfg = minimal_cfg()
    cfg["data"] = {"tickers": ["AMD"], "horizons": [1, 5]}
    cfg["training"] = {"impact_label_quantile": 0.8}
    baseline = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03"] * 2), "ticker": ["AMD"] * 2, "horizon": [1, 5],
        "fold_id": [1, 1], "analysis_split": ["train", "train"], "baseline_error": [0.1, 0.2],
        "actual_logvol": [-2.0, -1.9],
    })
    for column in STATE_COLUMNS:
        baseline[column] = 0.1
    baseline["baseline_prediction"] = 0.2
    events = pd.DataFrame({
        "row_id": [987], "date": pd.to_datetime(["2022-01-02"]),
        "effective_date": pd.to_datetime(["2022-01-03"]), "ticker": ["AMD"],
        "event_type": ["earnings"], "semantic_novelty": [0.5], "catalyst_score": [1.0],
        "entity_relevance": [1.0], "timestamp_confidence": [0.0], "word_count": [20],
        "embedding": [np.asarray([1.0, 0.0], dtype=np.float32)],
    })

    bundle = _tensor_bundle(baseline, events, cfg, torch.device("cpu"))

    assert bundle["edges"].row_id.tolist() == [987, 987]
    assert bundle["edges"].prediction_row_id.tolist() == [0, 1]
    assert bundle["edge_row"].tolist() == [0, 1]
    assert not {"row_id_x", "row_id_y", "date_x", "date_y"}.intersection(bundle["edges"].columns)

    output = {
        "correction": torch.zeros(2), "hurdle_probability": torch.zeros(2),
        "has_event": torch.ones(2, dtype=torch.bool), "edge_gate": torch.ones(2),
        "selected_mask": torch.ones(2, dtype=torch.bool),
    }
    predictions, edge_output = outputs_to_frames("T1_all_target", bundle, output)

    assert "prediction_row_id" not in predictions
    assert edge_output[["row_id", "prediction_row_id"]].values.tolist() == [[987, 0], [987, 1]]
