from __future__ import annotations

import pandas as pd


def add_relation_features(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    out["is_same_ticker_context"] = (
        out["context_ticker"].astype(str).eq(out["target_ticker"].astype(str)) & out["context_ticker"].astype(str).ne("")
    ).astype(float)
    out["is_direct_target"] = out["is_direct_target"].astype(float)
    out["static_graph_weight"] = out["static_graph_weight"].astype(float)
    out["static_graph_distance_clipped"] = out["static_graph_distance"].astype(float).clip(0, 10) / 10.0
    return out


def relation_feature_columns() -> list[str]:
    return ["is_direct_target", "is_same_ticker_context", "static_graph_weight", "static_graph_distance_clipped"]
