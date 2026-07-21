import pandas as pd

from src.stock_news_impact.abnormal_response import build_abnormal_response


def test_abnormal_response_actual_minus_stock_prediction():
    pred = pd.DataFrame(
        [
            {
                "model": "stock_only",
                "date": "2020-01-02",
                "target_date": "2020-01-03",
                "ticker": "ADI",
                "horizon": 1,
                "actual_logvol": -3.0,
                "final_prediction": -3.2,
                "split": "validation",
                "fold_id": 1,
                "seed": 42,
                "p_prediction": -3.1,
                "stock_residual_prediction": -0.1,
            }
        ]
    )
    out = build_abnormal_response(pred)
    assert abs(out["abnormal_volatility_response"].iloc[0] - 0.2) < 1e-9

