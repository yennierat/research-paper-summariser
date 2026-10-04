import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.db import connect
from src.ranker import LATEST_FEEDBACK, POSITIVE, active_model
from src.telegram import Telegram

log = logging.getLogger(__name__)

MIN_LABELS = 100
TEST_FRACTION = 0.2


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale, fit, then fold the scaling back in so the model is plain weights + bias."""
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    clf = LogisticRegression().fit((X - mean) / std, y)
    w = clf.coef_[0] / std
    return w, float(clf.intercept_[0] - w @ mean)


def evaluate(rows: list) -> tuple[str, tuple | None]:
    y = np.array([label in POSITIVE for _, label in rows], dtype=int)
    split = int(len(rows) * (1 - TEST_FRACTION))
    if len(rows) < MIN_LABELS or len(set(y[:split])) < 2 or len(set(y[split:])) < 2:
        return f"Retrainer: {len(rows)} usable labels, need {MIN_LABELS} with both likes and dislikes. Keeping current ranker.", None
    X = np.array([f for f, _ in rows])
    cw, cb = fit(X[:split], y[:split])
    return "", (X, y, split, cw, cb)


def retrain() -> None:
    with connect() as conn:
        rows = conn.execute(
            f"select p.features, f.label from ({LATEST_FEEDBACK}) f"
            " join papers p using (arxiv_id) where p.features is not null order by f.update_id"
        ).fetchall()
        msg, result = evaluate(rows)
        if result:
            X, y, split, cw, cb = result
            new_auc = roc_auc_score(y[split:], X[split:] @ cw + cb)
            ow, ob = active_model(conn)
            old_auc = roc_auc_score(y[split:], X[split:] @ ow + ob)
            promote = new_auc > old_auc
            w, b = fit(X, y)
            if promote:
                conn.execute("update models set active = false where active")
            conn.execute(
                "insert into models (weights, bias, auc, n_labels, active) values (%s, %s, %s, %s, %s)",
                (w.tolist(), b, new_auc, len(rows), promote),
            )
            msg = (f"Retrainer: {len(rows)} labels. Candidate AUC {new_auc:.3f} vs current {old_auc:.3f}. "
                   + ("Promoted new ranker." if promote else "Kept current ranker."))
    log.info(msg)
    Telegram().send_text(msg, parse_mode=None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retrain()
