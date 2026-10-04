import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src import retrain
from src.ranker import DEFAULT_WEIGHTS
from tests.fakes import FakeConn


def synthetic_rows(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                         rng.uniform(0, 6, n), rng.uniform(0, 3, n)])
    logit = 4 * X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2] - 2.5
    y = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    return [(list(x), "like" if v else "dislike") for x, v in zip(X, y)]


@pytest.fixture
def sent(monkeypatch):
    messages = []

    class FakeTelegram:
        def send_text(self, text, parse_mode="HTML"):
            messages.append(text)

    monkeypatch.setattr(retrain, "Telegram", FakeTelegram)
    return messages


def test_fit_folds_scaling_into_weights():
    rows = synthetic_rows()
    X = np.array([f for f, _ in rows])
    y = np.array([label == "like" for _, label in rows], dtype=int)
    w, b = retrain.fit(X, y)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    reference = LogisticRegression().fit((X - mean) / std, y).decision_function((X - mean) / std)
    assert np.allclose(X @ w + b, reference)


def test_save_counts_as_positive():
    rows = [([0.0] * 5, label) for label in ["save", "like", "dislike"] * 40]
    _, (X, y, split, w, b) = retrain.evaluate(rows)
    assert y.tolist()[:3] == [1, 1, 0]


def test_too_few_labels_keeps_current_ranker():
    msg, result = retrain.evaluate(synthetic_rows(n=50))
    assert result is None
    assert "need 100" in msg


def test_one_class_only_keeps_current_ranker():
    msg, result = retrain.evaluate([([0.0] * 5, "like")] * 150)
    assert result is None


def test_newest_rows_are_the_test_set():
    rows = synthetic_rows()
    _, (X, y, split, w, b) = retrain.evaluate(rows)
    assert split == 120
    assert X[split:].tolist() == [f for f, _ in rows[120:]]


def test_promotes_model_that_beats_hand_weights(monkeypatch, sent):
    conn = FakeConn(synthetic_rows())
    monkeypatch.setattr(retrain, "connect", lambda: conn)
    retrain.retrain()
    [(_, params)] = conn.statements("insert into models")
    assert params[4] is True
    assert conn.statements("update models set active = false")
    assert "Promoted" in sent[0]


def test_keeps_hand_weights_when_they_are_already_perfect(monkeypatch, sent):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (150, 5))
    score = X @ DEFAULT_WEIGHTS
    rows = [(list(x), "like" if s > np.median(score) else "dislike") for x, s in zip(X, score)]
    conn = FakeConn(rows)
    monkeypatch.setattr(retrain, "connect", lambda: conn)
    retrain.retrain()
    [(_, params)] = conn.statements("insert into models")
    assert params[4] is False
    assert not conn.statements("update models set active = false")
    assert "Kept current ranker" in sent[0]


def test_not_enough_labels_sends_message_and_saves_nothing(monkeypatch, sent):
    conn = FakeConn(synthetic_rows(n=20))
    monkeypatch.setattr(retrain, "connect", lambda: conn)
    retrain.retrain()
    assert not conn.statements("insert into models")
    assert "need 100" in sent[0]
