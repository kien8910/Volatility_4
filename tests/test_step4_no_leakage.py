import numpy as np

from src.graph.scalers import StandardScaler3D


def test_scaler_fit_train_only_not_validation():
    train = np.ones((2, 3, 4), dtype=np.float32)
    validation = np.full((1, 3, 4), 100.0, dtype=np.float32)
    all_x = np.concatenate([train, validation], axis=0)
    scaler = StandardScaler3D().fit(all_x[:2])
    transformed = scaler.transform(all_x)
    assert np.allclose(transformed[:2], 0.0)
    assert transformed[2].mean() > 90.0


def test_window_contains_only_history_through_origin():
    values = np.arange(10, dtype=np.float32)
    lookback = 4
    origin_idx = 6
    window = values[origin_idx - lookback + 1 : origin_idx + 1]
    assert window.tolist() == [3.0, 4.0, 5.0, 6.0]
    assert 7.0 not in window

