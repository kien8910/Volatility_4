from __future__ import annotations

import numpy as np
import pandas as pd

HIERARCHY_ORDER = ["macro", "sector", "target_company", "related_company", "filing"]
SCOPE_ORDER = ["macro", "sector", "firm_specific", "related_context", "filing_context"]


def add_event_features(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_text_log_chars"] = np.log1p(out["text_char_length"].astype(float))
    out["event_text_log_words"] = np.log1p(out["text_word_count"].astype(float))
    out["event_category_count"] = out["category_count"].astype(float)
    out["is_dynamic_news"] = out["is_dynamic_news"].astype(float)
    out["is_filing_context"] = out["is_filing_context"].astype(float)
    for hierarchy in HIERARCHY_ORDER:
        out[f"hierarchy_{hierarchy}"] = out["hierarchy"].astype(str).eq(hierarchy).astype(float)
    for scope in SCOPE_ORDER:
        out[f"scope_{scope}"] = out["event_scope"].astype(str).eq(scope).astype(float)
    return out


def event_feature_columns() -> list[str]:
    return (
        ["event_text_log_chars", "event_text_log_words", "event_category_count", "is_dynamic_news", "is_filing_context"]
        + [f"hierarchy_{h}" for h in HIERARCHY_ORDER]
        + [f"scope_{s}" for s in SCOPE_ORDER]
    )
