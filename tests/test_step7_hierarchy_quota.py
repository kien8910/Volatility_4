import pandas as pd

from src.stock_news_impact.run_step7 import mode_build_events


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


def test_hierarchy_quota_keeps_target_company_and_sector(tmp_path, monkeypatch):
    rows = []
    for hierarchy in ["macro", "sector", "target_company", "related_company"]:
        for i in range(4):
            rows.append(
                {
                    "event_id": f"{hierarchy}-{i}",
                    "date": pd.Timestamp("2020-01-02"),
                    "hierarchy": hierarchy,
                    "source_ticker": "ADI" if hierarchy in {"target_company", "related_company"} else "",
                    "context_ticker": "ADI" if hierarchy in {"target_company", "related_company"} else "",
                    "text": f"{hierarchy} text {i}",
                    "text_hash": f"h-{hierarchy}-{i}",
                    "has_text": 1,
                    "event_scope": "firm_specific" if hierarchy == "target_company" else hierarchy,
                    "is_dynamic_news": 1,
                    "is_filing_context": 0,
                    "category_count": 1,
                    "text_char_length": 10,
                    "text_word_count": 2,
                }
            )

    monkeypatch.setattr("src.stock_news_impact.run_step7.load_news_features_for_step7", lambda cfg: pd.DataFrame())
    monkeypatch.setattr("src.stock_news_impact.run_step7.build_news_events", lambda features, cfg: pd.DataFrame(rows))
    monkeypatch.setattr("src.stock_news_impact.run_step7.atomic_parquet", lambda df, path: None)
    cfg = {
        "experiment": {"output_dir": str(tmp_path)},
        "pilot": {"max_events_per_date_by_hierarchy": {"macro": 1, "sector": 1, "target_company": 2, "related_company": 1}},
    }
    events = mode_build_events(cfg, DummyLogger())
    counts = events.groupby("hierarchy").size().to_dict()
    assert counts["macro"] == 1
    assert counts["sector"] == 1
    assert counts["target_company"] == 2
    assert counts["related_company"] == 1

