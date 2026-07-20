import pandas as pd

from src.decomposition.walk_forward_predictor import build_segments


def test_splits_have_no_wrong_overlap():
    dates = pd.bdate_range("2020-01-01", periods=20)
    split = pd.DataFrame({"date": dates, "is_locked_test": [0] * 15 + [1] * 5, "base_split": ["development"] * 15 + ["locked_test"] * 5})
    folds = pd.DataFrame({"fold_id": [1] * 3, "date": dates[10:13], "role": ["validation"] * 3})
    segments = build_segments(pd.DataFrame(), split, folds, initial_training_days=5)
    val_dates = set(segments[1].origin_dates)
    test_dates = set(segments[-1].origin_dates)
    assert not val_dates & test_dates
