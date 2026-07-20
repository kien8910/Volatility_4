import pandas as pd


def test_target_date_greater_than_forecast_origin_rule():
    targets = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "target_date": pd.to_datetime(["2020-01-02", "2020-01-08"]),
    })
    assert targets["target_date"].gt(targets["date"]).all()


def test_training_target_date_not_after_origin_rule():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-10", "2020-01-11"]),
        "max_training_target_date": pd.to_datetime(["2020-01-10", "2020-01-09"]),
    })
    assert rows["max_training_target_date"].le(rows["date"]).all()
