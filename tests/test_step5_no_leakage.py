import numpy as np

from src.regime_graph.state_features import market_state_labels


def test_market_state_thresholds_train_only():
    import pandas as pd

    frame = pd.DataFrame({"market_logvol_mean": [1.0, 1.0, 1.0, 100.0], "split": ["train", "train", "train", "validation"]})
    labels = market_state_labels(frame, np.array([0, 1, 2]), 0.33, 0.67)
    assert labels[3] == "high_volatility"

