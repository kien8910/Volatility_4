import pandas as pd

from src.stock_news_impact.utility_labels import add_utility_labels


def test_utility_labels_train_margin_only():
    frame = pd.DataFrame(
        {
            "actual_logvol": [1.0, 1.0, 1.0, 1.0],
            "stock_prediction": [0.0, 0.0, 0.0, 0.0],
            "news_correction_proxy": [0.95, -1.0, 0.05, 0.0],
        }
    )
    out = add_utility_labels(frame, pd.Series([True, True, True, False]), margin_quantile=0.0)
    assert out["utility_label"].iloc[0] == 1
    assert out["utility_label"].iloc[1] == 0
