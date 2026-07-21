from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.graph import SEMICONDUCTOR_TICKERS

HIERARCHY_COLUMNS: dict[str, list[str]] = {
    "macro": [f"macro_category{i}" for i in range(1, 6)],
    "sector": [f"sector_category{i}" for i in range(1, 6)],
    "target_company": [f"targetCompany_category{i}" for i in range(1, 4)],
    "related_company": [f"relatedCompany_category{i}" for i in range(1, 4)],
    "filing": [
        "filing_financialStatement",
        "filing_governanceRisks",
        "filing_overviewProduct",
        "filing_recentEventCatalyst",
        "filing_strategyMarketOps",
    ],
}

TEXT_COLUMNS = [f"{name}_text" for name in HIERARCHY_COLUMNS]
MISSING_TOKENS = {"", "none", "nan", "null"}


class Step6NewsDataError(ValueError):
    """Raised when Step 6 news data fail an explicit leakage/schema check."""


def normalize_text(value: object) -> tuple[str, int]:
    if value is None:
        return "", 1
    if isinstance(value, float) and np.isnan(value):
        return "", 1
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return "", 1
    return " ".join(text.split()), 0


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _join_category_texts(row: pd.Series, columns: list[str]) -> tuple[str, int, int]:
    parts: list[str] = []
    missing = 0
    present = 0
    for idx, col in enumerate(columns, start=1):
        normalized, is_missing = normalize_text(row[col] if col in row else "")
        missing += is_missing
        if not is_missing:
            present += 1
            parts.append(f"[CATEGORY_{idx}] {normalized}")
    return "\n".join(parts), int(missing == len(columns)), present


def _require_base_columns(df: pd.DataFrame) -> None:
    missing = [col for col in ["date", "ticker"] if col not in df.columns]
    if missing:
        raise Step6NewsDataError(f"News panel is missing required columns: {missing}")


def build_hierarchy_features_from_panel(panel: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or SEMICONDUCTOR_TICKERS
    _require_base_columns(panel)
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].isin(tickers)].copy()
    if sorted(df["ticker"].unique().tolist()) != sorted(tickers):
        raise Step6NewsDataError("Panel must contain exactly the configured semiconductor tickers.")
    if df.duplicated(["date", "ticker"]).any():
        dup = df.loc[df.duplicated(["date", "ticker"], keep=False), ["date", "ticker"]].head()
        raise Step6NewsDataError(f"Duplicate date x ticker news rows:\n{dup}")

    rows = []
    for row in df.itertuples(index=False):
        record = {"date": pd.Timestamp(getattr(row, "date")), "ticker": getattr(row, "ticker")}
        series = pd.Series(row._asdict())
        for hierarchy, columns in HIERARCHY_COLUMNS.items():
            text, missing_mask, count = _join_category_texts(series, columns)
            prefix = hierarchy
            record[f"{prefix}_text"] = text
            record[f"{prefix}_missing_mask"] = missing_mask
            record[f"has_{prefix}"] = int(count > 0)
            record[f"{prefix}_count"] = int(count)
            record[f"{prefix}_text_hash"] = stable_text_hash(text)
            record[f"{prefix}_text_char_length"] = int(len(text))
            record[f"{prefix}_text_word_count"] = int(len(text.split())) if text else 0
        rows.append(record)
    out = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    out["has_any_dynamic_news"] = (
        out[["has_macro", "has_sector", "has_target_company", "has_related_company"]].sum(axis=1) > 0
    ).astype(int)
    validate_news_feature_frame(out, tickers)
    return out


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Step6NewsDataError(f"Unsupported news input format: {path}")


def load_or_build_news_features(panel_path: str | Path, news_long_path: str | Path, tickers: list[str]) -> pd.DataFrame:
    news_long_path = Path(news_long_path)
    panel_path = Path(panel_path)
    if news_long_path.exists():
        long_df = _read_table(news_long_path)
        try:
            return build_hierarchy_features_from_long(long_df, tickers)
        except Step6NewsDataError:
            if not panel_path.exists():
                raise
    if not panel_path.exists():
        raise Step6NewsDataError(f"Missing both news_long_path and panel_path: {news_long_path}, {panel_path}")
    return build_hierarchy_features_from_panel(_read_table(panel_path), tickers)


def build_hierarchy_features_from_long(news_long: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or SEMICONDUCTOR_TICKERS
    required = {"date", "ticker", "hierarchy", "category", "text"}
    missing = sorted(required - set(news_long.columns))
    if missing:
        raise Step6NewsDataError(f"Long news table is missing required columns: {missing}")
    df = news_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].isin(tickers)].copy()
    rows = []
    for (date, ticker), grp in df.groupby(["date", "ticker"], sort=True):
        record = {"date": pd.Timestamp(date), "ticker": ticker}
        for hierarchy, columns in HIERARCHY_COLUMNS.items():
            parts = []
            for pos, category in enumerate(columns, start=1):
                match = grp.loc[
                    (grp["hierarchy"].astype(str) == hierarchy)
                    & (grp["category"].astype(str).isin([category, f"CATEGORY_{pos}", str(pos)]))
                ]
                text, missing = normalize_text(" ".join(match["text"].dropna().astype(str).tolist()))
                if not missing:
                    parts.append(f"[CATEGORY_{pos}] {text}")
            full = "\n".join(parts)
            record[f"{hierarchy}_text"] = full
            record[f"{hierarchy}_missing_mask"] = int(not bool(full))
            record[f"has_{hierarchy}"] = int(bool(full))
            record[f"{hierarchy}_count"] = len(parts)
            record[f"{hierarchy}_text_hash"] = stable_text_hash(full)
            record[f"{hierarchy}_text_char_length"] = len(full)
            record[f"{hierarchy}_text_word_count"] = len(full.split()) if full else 0
        rows.append(record)
    out = pd.DataFrame(rows)
    full_index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(sorted(df["date"].unique())), tickers], names=["date", "ticker"]
    )
    out = out.set_index(["date", "ticker"]).reindex(full_index).reset_index()
    for hierarchy in HIERARCHY_COLUMNS:
        out[f"{hierarchy}_text"] = out[f"{hierarchy}_text"].fillna("")
        out[f"{hierarchy}_missing_mask"] = out[f"{hierarchy}_missing_mask"].fillna(1).astype(int)
        out[f"has_{hierarchy}"] = out[f"has_{hierarchy}"].fillna(0).astype(int)
        out[f"{hierarchy}_count"] = out[f"{hierarchy}_count"].fillna(0).astype(int)
        out[f"{hierarchy}_text_hash"] = out[f"{hierarchy}_text"].map(stable_text_hash)
        out[f"{hierarchy}_text_char_length"] = out[f"{hierarchy}_text"].str.len().astype(int)
        out[f"{hierarchy}_text_word_count"] = out[f"{hierarchy}_text"].map(lambda x: len(x.split()) if x else 0)
    out["has_any_dynamic_news"] = (
        out[["has_macro", "has_sector", "has_target_company", "has_related_company"]].sum(axis=1) > 0
    ).astype(int)
    validate_news_feature_frame(out, tickers)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def embedding_request_frame(features: pd.DataFrame, hierarchies: Iterable[str]) -> pd.DataFrame:
    rows = []
    for hierarchy in hierarchies:
        rows.append(
            features[
                [f"{hierarchy}_text", f"{hierarchy}_text_hash", f"{hierarchy}_missing_mask"]
            ].rename(
                columns={
                    f"{hierarchy}_text": "text",
                    f"{hierarchy}_text_hash": "text_hash",
                    f"{hierarchy}_missing_mask": "missing_mask",
                }
            ).assign(hierarchy=hierarchy)
        )
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["hierarchy", "text_hash"]).reset_index(drop=True)


def validate_news_feature_frame(features: pd.DataFrame, tickers: list[str]) -> None:
    required = ["date", "ticker", "has_any_dynamic_news"]
    for hierarchy in HIERARCHY_COLUMNS:
        required.extend(
            [
                f"{hierarchy}_text",
                f"{hierarchy}_text_hash",
                f"{hierarchy}_missing_mask",
                f"has_{hierarchy}",
                f"{hierarchy}_count",
                f"{hierarchy}_text_char_length",
                f"{hierarchy}_text_word_count",
            ]
        )
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise Step6NewsDataError(f"News feature frame is missing columns: {missing}")
    if sorted(features["ticker"].unique().tolist()) != sorted(tickers):
        raise Step6NewsDataError("News feature frame does not match configured ticker universe.")
    if features.duplicated(["date", "ticker"]).any():
        raise Step6NewsDataError("News feature frame has duplicate date x ticker rows.")
