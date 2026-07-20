import numpy as np

from src.graph.metrics import qlike_from_logvol


def test_qlike_zero_when_prediction_equals_actual():
    actual = np.array([0.1, -0.2, 0.3])
    qlike, clipped = qlike_from_logvol(actual, actual)
    assert np.allclose(qlike, 0.0)
    assert clipped == 0


def test_qlike_clips_extreme_logvol():
    actual = np.array([100.0])
    pred = np.array([-100.0])
    qlike, clipped = qlike_from_logvol(actual, pred, clip_min=-20.0, clip_max=20.0)
    assert np.isfinite(qlike).all()
    assert clipped == 2

