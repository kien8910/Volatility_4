from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS
from src.news.text_preprocessing import HIERARCHY_COLUMNS, load_or_build_news_features


class Step7DataError(ValueError):
    """Raised when Step 7 data fail an explicit schema or leakage check."""


def stable_event_id(*parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_news_features_for_step7(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data"].get("step6_news_features_path", ""))
    tickers = list(cfg["data"]["tickers"])
    if path.exists():
        features = pd.read_parquet(path)
    else:
        features = load_or_build_news_features(cfg["data"]["panel_path"], cfg["data"]["news_long_path"], tickers)
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])
    if sorted(features["ticker"].unique().tolist()) != sorted(tickers):
        raise Step7DataError("Step 7 news features do not match the 11 semiconductor tickers.")
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_news_events(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    include = set(cfg.get("events", {}).get("include_hierarchies") or HIERARCHY_COLUMNS.keys())
    rows: list[dict] = []
    for hierarchy in ["macro", "sector"]:
        if hierarchy not in include:
            continue
        sub = features.loc[features[f"has_{hierarchy}"].astype(int) == 1].copy()
        sub = sub.drop_duplicates(["date", f"{hierarchy}_text_hash"])
        for row in sub.itertuples(index=False):
            text = getattr(row, f"{hierarchy}_text")
            text_hash = getattr(row, f"{hierarchy}_text_hash")
            rows.append(
                {
                    "event_id": stable_event_id(hierarchy, row.date, text_hash),
                    "date": pd.Timestamp(row.date),
                    "hierarchy": hierarchy,
                    "source_ticker": "",
                    "context_ticker": "",
                    "text": text,
                    "text_hash": text_hash,
                    "has_text": 1,
                    "event_scope": hierarchy,
                    "is_dynamic_news": 1,
                    "is_filing_context": 0,
                    "category_count": int(getattr(row, f"{hierarchy}_count")),
                    "text_char_length": int(getattr(row, f"{hierarchy}_text_char_length")),
                    "text_word_count": int(getattr(row, f"{hierarchy}_text_word_count")),
                }
            )
    for hierarchy, scope in [("target_company", "firm_specific"), ("related_company", "related_context")]:
        if hierarchy not in include:
            continue
        sub = features.loc[features[f"has_{hierarchy}"].astype(int) == 1].copy()
        for row in sub.itertuples(index=False):
            text = getattr(row, f"{hierarchy}_text")
            text_hash = getattr(row, f"{hierarchy}_text_hash")
            ticker = str(row.ticker)
            rows.append(
                {
                    "event_id": stable_event_id(hierarchy, row.date, ticker, text_hash),
                    "date": pd.Timestamp(row.date),
                    "hierarchy": hierarchy,
                    "source_ticker": ticker,
                    "context_ticker": ticker,
                    "text": text,
                    "text_hash": text_hash,
                    "has_text": 1,
                    "event_scope": scope,
                    "is_dynamic_news": 1,
                    "is_filing_context": 0,
                    "category_count": int(getattr(row, f"{hierarchy}_count")),
                    "text_char_length": int(getattr(row, f"{hierarchy}_text_char_length")),
                    "text_word_count": int(getattr(row, f"{hierarchy}_text_word_count")),
                }
            )
    if "filing" in include and bool(cfg.get("events", {}).get("treat_filing_as_context", True)):
        sub = features.loc[features["has_filing"].astype(int) == 1].copy()
        sub = sub.sort_values("date").drop_duplicates(["ticker", "filing_text_hash"], keep="first")
        for row in sub.itertuples(index=False):
            ticker = str(row.ticker)
            rows.append(
                {
                    "event_id": stable_event_id("filing", ticker, row.filing_text_hash),
                    "date": pd.Timestamp(row.date),
                    "hierarchy": "filing",
                    "source_ticker": ticker,
                    "context_ticker": ticker,
                    "text": row.filing_text,
                    "text_hash": row.filing_text_hash,
                    "has_text": 1,
                    "event_scope": "filing_context",
                    "is_dynamic_news": 0,
                    "is_filing_context": 1,
                    "category_count": int(row.filing_count),
                    "text_char_length": int(row.filing_text_char_length),
                    "text_word_count": int(row.filing_text_word_count),
                }
            )
    events = pd.DataFrame(rows)
    if events.empty:
        raise Step7DataError("No Step 7 news events were built.")
    events = events.drop_duplicates("event_id").sort_values(["date", "hierarchy", "event_id"]).reset_index(drop=True)
    return events


def validate_events(events: pd.DataFrame) -> None:
    required = [
        "event_id",
        "date",
        "hierarchy",
        "source_ticker",
        "context_ticker",
        "text",
        "text_hash",
        "has_text",
        "event_scope",
        "is_dynamic_news",
        "is_filing_context",
    ]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise Step7DataError(f"Step 7 event table missing columns: {missing}")
    if events["event_id"].duplicated().any():
        raise Step7DataError("Step 7 event_id must be unique.")
    bad_tickers = set(events.loc[events["context_ticker"].astype(str) != "", "context_ticker"]) - set(SEMICONDUCTOR_TICKERS)
    if bad_tickers:
        raise Step7DataError(f"Unexpected context tickers in event table: {sorted(bad_tickers)}")
