from __future__ import annotations

import re
import numpy as np
import pandas as pd

DEFAULT_CATALYSTS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "revenue", "profit", "eps", "quarter", "fiscal"),
    "guidance": ("guidance", "forecast", "outlook", "expects", "estimate", "revision"),
    "regulatory": ("regulator", "regulatory", "antitrust", "investigation", "approval", "sanction"),
    "ma": ("acquire", "acquisition", "merger", "takeover", "deal"),
    "supply_chain": ("supply", "shortage", "shipment", "capacity", "factory", "fab", "export control"),
    "product": ("launch", "product", "chip", "processor", "gpu", "delay", "roadmap"),
    "analyst": ("upgrade", "downgrade", "price target", "rating", "analyst"),
    "management": ("ceo", "cfo", "resign", "appoint", "management", "executive"),
}


def classify_event_type(text: str, catalyst_keywords: dict[str, list[str] | tuple[str, ...]]) -> tuple[str, float]:
    normalized = " ".join(str(text).lower().split())
    for event_type, keywords in catalyst_keywords.items():
        if any(re.search(rf"\b{re.escape(str(keyword).lower())}\b", normalized) for keyword in keywords):
            return str(event_type), 1.0
    return "other", 0.0


def hard_filter_events(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    required = ["event_id", "news_date", "effective_date", "ticker", "text", "text_hash", "category"]
    missing = [c for c in required if c not in events]
    if missing:
        raise ValueError(f"Event frame missing columns: {missing}")
    out = events.copy()
    keywords = cfg["event_filter"].get("catalyst_keywords", DEFAULT_CATALYSTS)
    classified = out.text.map(lambda x: classify_event_type(str(x), keywords))
    out["event_type"] = classified.map(lambda x: x[0])
    out["catalyst_score"] = classified.map(lambda x: float(x[1]))
    out["word_count"] = out.text.fillna("").astype(str).str.split().str.len().fillna(0).astype(int)
    out["entity_relevance"] = 1.0  # target_company hierarchy is a direct assignment in FinTexTS.
    out["timestamp_confidence"] = np.where(out.get("publication_timestamp", pd.Series(index=out.index)).notna(), 1.0, 0.0)
    out["basic_filter_pass"] = (
        out.text.fillna("").astype(str).str.strip().ne("")
        & out.word_count.ge(int(cfg["event_filter"].get("min_words", 8)))
        & out.word_count.le(int(cfg["event_filter"].get("max_words", 512)))
    )
    out["hard_filter_pass"] = out.basic_filter_pass.copy()
    if bool(cfg["event_filter"].get("require_catalyst_keyword", True)):
        out["hard_filter_pass"] &= out.catalyst_score.gt(0)
    out = out.sort_values(["effective_date", "ticker", "event_id"]).drop_duplicates(
        ["effective_date", "ticker", "text_hash"], keep="first"
    )
    return out.reset_index(drop=True)


def deterministic_selection_score(events: pd.DataFrame) -> pd.Series:
    novelty_source = (events["semantic_novelty"] if "semantic_novelty" in events
                      else pd.Series(0.0, index=events.index, dtype=float))
    novelty = pd.to_numeric(novelty_source, errors="coerce").fillna(0.0).clip(0, 2) / 2.0
    return (
        0.35 * events.entity_relevance.astype(float)
        + 0.25 * events.catalyst_score.astype(float)
        + 0.25 * novelty
        + 0.15 * events.timestamp_confidence.astype(float)
    )


def deterministic_top_k(events: pd.DataFrame, k: int) -> pd.DataFrame:
    out = events.loc[events.hard_filter_pass].copy()
    out["deterministic_score"] = deterministic_selection_score(out)
    return (out.sort_values(["effective_date", "ticker", "deterministic_score", "event_id"],
                            ascending=[True, True, False, True])
            .groupby(["effective_date", "ticker"], sort=False).head(int(k)).reset_index(drop=True))
