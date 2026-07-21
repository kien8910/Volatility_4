import pandas as pd

from src.stock_news_impact.event_features_builder import build_gate_feature_frame


def test_gate_feature_builder_drops_correction_ticker_column():
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "date": pd.Timestamp("2020-01-02"),
                "hierarchy": "target_company",
                "source_ticker": "ADI",
                "context_ticker": "ADI",
                "text": "ADI event",
                "text_hash": "h",
                "has_text": 1,
                "event_scope": "firm_specific",
                "is_dynamic_news": 1,
                "is_filing_context": 0,
                "category_count": 1,
                "text_char_length": 9,
                "text_word_count": 2,
            }
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "date": pd.Timestamp("2020-01-02"),
                "target_date": pd.Timestamp("2020-01-03"),
                "source_ticker": "ADI",
                "target_ticker": "ADI",
                "context_ticker": "ADI",
                "hierarchy": "target_company",
                "event_scope": "firm_specific",
                "horizon": 1,
                "split": "validation",
                "fold_id": 1,
                "seed": 42,
                "is_direct_target": 1,
                "static_graph_weight": 1.0,
                "static_graph_distance": 0,
                "placebo_type": "real",
            }
        ]
    )
    step6 = pd.DataFrame(
        [
            {
                "model": "stock_only",
                "config_id": "s0",
                "date": pd.Timestamp("2020-01-02"),
                "target_date": pd.Timestamp("2020-01-03"),
                "ticker": "ADI",
                "horizon": 1,
                "actual_logvol": -3.0,
                "final_prediction": -3.1,
                "split": "validation",
                "fold_id": 1,
                "seed": 42,
                "p_prediction": -3.0,
                "stock_residual_prediction": -0.1,
                "news_residual_correction": 0.0,
            },
            {
                "model": "concatenation",
                "config_id": "news",
                "date": pd.Timestamp("2020-01-02"),
                "target_date": pd.Timestamp("2020-01-03"),
                "ticker": "ADI",
                "horizon": 1,
                "actual_logvol": -3.0,
                "final_prediction": -3.05,
                "split": "validation",
                "fold_id": 1,
                "seed": 42,
                "p_prediction": -3.0,
                "stock_residual_prediction": -0.1,
                "news_residual_correction": 0.05,
            },
        ]
    )
    cfg = {"utility_supervision": {"margin_quantiles": [0.0]}}
    frame, cols = build_gate_feature_frame(events, pairs, step6, {"config_id": "news"}, cfg)
    assert "ticker" not in frame.columns
    assert "target_ticker" in frame.columns
    assert frame["ticker_code"].iloc[0] == 0.0
