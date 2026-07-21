import pandas as pd

from src.stock_news_impact.treatment_control import simple_treatment_control_did


def test_treatment_control_did_outputs_rows():
    frame = pd.DataFrame(
        [
            {"date": "2020-01-02", "horizon": 1, "hierarchy": "target_company", "is_direct_target": 1, "abnormal_volatility_response": 2.0},
            {"date": "2020-01-02", "horizon": 1, "hierarchy": "target_company", "is_direct_target": 0, "abnormal_volatility_response": 1.0},
        ]
    )
    out = simple_treatment_control_did(frame)
    assert out["mean_did"].iloc[0] == 1.0

