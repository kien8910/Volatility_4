import pandas as pd

from src.regime_graph.run_step5 import GRAPH_DIVERSITY_COLUMNS, append_table


def test_append_table_handles_empty_csv_file(tmp_path):
    path = tmp_path / "graph_diversity.csv"
    path.write_text("", encoding="utf-8")
    incoming = pd.DataFrame(
        [
            {
                "config_id": "x",
                "model": "S5-R",
                "fold_id": 1,
                "seed": 42,
                "regime_a": 1,
                "regime_b": 2,
                "frobenius_distance": 0.1,
                "cosine_similarity": 0.9,
                "spearman_correlation": 0.8,
                "topk_jaccard": 0.5,
            }
        ]
    )
    merged = append_table(
        incoming,
        path,
        ["config_id", "fold_id", "seed", "regime_a", "regime_b"],
        parquet=False,
        columns=GRAPH_DIVERSITY_COLUMNS,
    )
    assert len(merged) == 1
    assert pd.read_csv(path).shape[0] == 1


def test_append_table_skips_no_column_empty_dataframe_without_creating_bad_csv(tmp_path):
    path = tmp_path / "graph_diversity.csv"
    merged = append_table(
        pd.DataFrame(),
        path,
        ["config_id", "fold_id", "seed", "regime_a", "regime_b"],
        parquet=False,
        columns=GRAPH_DIVERSITY_COLUMNS,
    )
    assert list(merged.columns) == GRAPH_DIVERSITY_COLUMNS
    assert pd.read_csv(path).empty
