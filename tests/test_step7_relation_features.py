import pandas as pd

from src.stock_news_impact.relation_features import add_relation_features, relation_feature_columns


def test_relation_features_direct_context():
    frame = pd.DataFrame([{"context_ticker": "ADI", "target_ticker": "ADI", "is_direct_target": 1, "static_graph_weight": 0.5, "static_graph_distance": 0}])
    out = add_relation_features(frame)
    assert out["is_same_ticker_context"].iloc[0] == 1.0
    assert set(relation_feature_columns()).issubset(out.columns)

