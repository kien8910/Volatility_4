from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.graph.datasets import GraphWindowDataset
from src.news import NEWS_HIERARCHIES
from src.news.embedding_cache import embedding_lookup


CONTROL_FEATURES = []
for _h in NEWS_HIERARCHIES:
    CONTROL_FEATURES.extend([f"has_{_h}", f"{_h}_count", f"{_h}_missing_mask", f"{_h}_text_char_length", f"{_h}_text_word_count"])
CONTROL_FEATURES.append("has_any_dynamic_news")


@dataclass(frozen=True)
class AlignedNewsTensors:
    embeddings: np.ndarray
    controls: np.ndarray
    coverage: pd.DataFrame
    hierarchies: list[str]
    control_columns: list[str]


def align_news_to_samples(
    samples,
    features: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    encoder_name: str,
    pooling_method: str,
    max_length: int,
    hierarchies: list[str] | None = None,
) -> AlignedNewsTensors:
    hierarchies = hierarchies or list(NEWS_HIERARCHIES)
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])
    lookup = embedding_lookup(embedding_frame, encoder_name, pooling_method, max_length)
    if lookup:
        emb_dim = len(next(iter(lookup.values())))
    else:
        emb_dim = int(embedding_frame["embedding_dim"].dropna().iloc[0]) if "embedding_dim" in embedding_frame and not embedding_frame.empty else 0
    if emb_dim <= 0:
        raise ValueError("Cannot infer embedding dimension from embedding cache.")
    index = pd.MultiIndex.from_product([samples.sample_dates, samples.tickers], names=["date", "ticker"])
    aligned = features.set_index(["date", "ticker"]).reindex(index).reset_index()
    for hierarchy in hierarchies:
        for col, fill in [
            (f"{hierarchy}_text_hash", ""),
            (f"{hierarchy}_missing_mask", 1),
            (f"has_{hierarchy}", 0),
            (f"{hierarchy}_count", 0),
            (f"{hierarchy}_text_char_length", 0),
            (f"{hierarchy}_text_word_count", 0),
        ]:
            aligned[col] = aligned[col].fillna(fill)
    aligned["has_any_dynamic_news"] = aligned["has_any_dynamic_news"].fillna(0).astype(int)
    n_samples, n_nodes, n_h = len(samples.sample_dates), len(samples.tickers), len(hierarchies)
    embeddings = np.zeros((n_samples, n_nodes, n_h, emb_dim), dtype=np.float32)
    for sample_pos in range(n_samples):
        offset = sample_pos * n_nodes
        rows = aligned.iloc[offset : offset + n_nodes]
        for h_idx, hierarchy in enumerate(hierarchies):
            for node_idx, text_hash in enumerate(rows[f"{hierarchy}_text_hash"].astype(str).tolist()):
                embeddings[sample_pos, node_idx, h_idx] = lookup.get(text_hash, np.zeros(emb_dim, dtype=np.float32))
    controls = aligned[CONTROL_FEATURES].fillna(0).to_numpy(dtype=np.float32).reshape(n_samples, n_nodes, len(CONTROL_FEATURES))
    coverage = aligned[["date", "ticker"] + CONTROL_FEATURES].copy()
    return AlignedNewsTensors(embeddings, controls, coverage, hierarchies, CONTROL_FEATURES)


class Step6WindowDataset(Dataset):
    def __init__(self, base: GraphWindowDataset, news: AlignedNewsTensors, ablation_mask: np.ndarray | None = None):
        self.base = base
        self.news = news
        self.source_indices = base.indices
        if ablation_mask is None:
            ablation_mask = np.ones(len(news.hierarchies), dtype=np.float32)
        self.ablation_mask = np.asarray(ablation_mask, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        source_idx = int(item["sample_index"])
        emb = self.news.embeddings[source_idx] * self.ablation_mask[None, :, None]
        controls = self.news.controls[source_idx].copy()
        return {
            **item,
            "news_embeddings": torch.from_numpy(emb).float(),
            "news_controls": torch.from_numpy(controls).float(),
        }


def ablation_mask(name: str, hierarchies: list[str]) -> np.ndarray:
    groups = {
        "all_dynamic": {"macro", "sector", "target_company", "related_company"},
        "all_including_filing": set(hierarchies),
        "macro_only": {"macro"},
        "sector_only": {"sector"},
        "target_company_only": {"target_company"},
        "related_company_only": {"related_company"},
        "filing_only": {"filing"},
    }
    if name not in groups:
        raise ValueError(f"Unknown Step 6 hierarchy ablation: {name}")
    return np.asarray([1.0 if h in groups[name] else 0.0 for h in hierarchies], dtype=np.float32)

