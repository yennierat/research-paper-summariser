import math
import numpy as np
import pytest
from src.ranker import build_features, active_model, DEFAULT_WEIGHTS, FEATURES


class OneRowConn:
    def __init__(self, row):
        self.row = row

    def execute(self, *a):
        return self

    def fetchone(self):
        return self.row


def test_build_features():
    f = build_features(np.array([1.0, 0.0]), np.array([0.6, 0.8]), np.array([1.0, 0.0]),
                       np.array([0.0, 1.0]), 9, 2.5)
    assert f == pytest.approx([0.6, 1.0, 0.0, math.log(10), 2.5])
    assert len(f) == len(FEATURES) == len(DEFAULT_WEIGHTS)


def test_build_features_before_any_feedback():
    f = build_features(np.array([1.0, 0.0]), np.array([1.0, 0.0]), None, None, None, 0.0)
    assert f == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


def test_active_model_falls_back_to_hand_weights():
    w, b = active_model(OneRowConn(None))
    assert np.array_equal(w, DEFAULT_WEIGHTS)
    assert b == 0.0


def test_active_model_uses_stored_weights():
    w, b = active_model(OneRowConn(([1, 2, 3, 4, 5], 0.5)))
    assert w.tolist() == [1, 2, 3, 4, 5]
    assert b == 0.5
