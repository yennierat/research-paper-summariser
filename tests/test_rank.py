from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

pytest.importorskip("sentence_transformers")
from src import rank  # noqa: E402

NOW = datetime.now(timezone.utc)
VECTORS = {
    "close": [1.0, 0.0],
    "medium": [0.8, 0.6],
    "far": [0.0, 1.0],
}


class FakeModel:
    def __init__(self, name):
        pass

    def encode(self, texts, normalize_embeddings):
        if isinstance(texts, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([VECTORS[t.split(".")[0]] for t in texts], dtype=np.float32)


class RankConn:
    def __init__(self, candidates, rated=()):
        self.candidates, self.rated, self.updates = candidates, list(rated), []

    def execute(self, sql, params=None):
        self.last = sql
        if sql.startswith("update"):
            self.updates.append(params)
        return self

    def fetchall(self):
        if "selected_at is null" in self.last:
            return self.candidates
        return self.rated if "join papers p using" in self.last else []

    def fetchone(self):
        return None

    def commit(self):
        pass


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(rank, "SentenceTransformer", FakeModel)


def candidate(name, categories=("cs.LG",)):
    return (f"id-{name}", name, "abstract", None, NOW - timedelta(days=1), list(categories))


def test_rank_returns_best_first():
    conn = RankConn([candidate("far"), candidate("close"), candidate("medium")])
    assert rank.rank(conn, n=2) == ["id-close", "id-medium"]


def test_rank_spreads_picks_across_categories():
    conn = RankConn([candidate("close"), candidate("medium"), candidate("far", ["cs.CR"])])
    assert rank.rank(conn, n=2) == ["id-close", "id-far"]


def test_pick_takes_best_paper_per_category():
    cats = [["cs.LG"], ["cs.LG"], ["cs.CL"], ["cs.AI"], ["cs.CV"], ["cs.CR"], ["cs.CR"]]
    assert rank.pick(list(range(7)), cats, 5) == [0, 2, 3, 4, 5]


def test_pick_does_not_pick_a_cross_listed_paper_twice():
    cats = [["cs.LG", "cs.CL"], ["cs.LG"], ["cs.CL"]]
    assert rank.pick([0, 1, 2], cats, 5) == [0, 1, 2]


def test_pick_fills_empty_categories_from_overall_order():
    cats = [["cs.LG"], [], ["cs.LG"], ["cs.CV"], [], []]
    assert rank.pick([0, 1, 2, 3, 4, 5], cats, 5) == [0, 1, 2, 3, 4]


def test_pick_returns_results_in_score_order():
    cats = [["cs.CR"], ["cs.LG"], ["cs.CL"]]
    assert rank.pick([2, 0, 1], cats, 3) == [2, 0, 1]


def test_rank_stores_features_used_for_training():
    conn = RankConn([candidate("close")])
    rank.rank(conn, n=1)
    [(embedding, features, arxiv_id)] = conn.updates
    assert arxiv_id == "id-close"
    assert embedding == pytest.approx([1.0, 0.0])
    assert features[0] == pytest.approx(1.0)
    assert features[4] == pytest.approx(1.0, abs=0.01)


def stored_features(rated):
    conn = RankConn([candidate("close"), candidate("far")], rated)
    rank.rank(conn, n=2)
    return {arxiv_id: features for _, features, arxiv_id in conn.updates}


def test_likes_and_dislikes_shape_features():
    f = stored_features([([1.0, 0.0], "like"), ([0.0, 1.0], "dislike")])
    assert f["id-close"][1:3] == pytest.approx([1.0, 0.0])
    assert f["id-far"][1:3] == pytest.approx([0.0, 1.0])


def test_save_counts_as_liked():
    f = stored_features([([0.0, 1.0], "save")])
    assert f["id-far"][1:3] == pytest.approx([1.0, 0.0])


def test_liked_centroid_is_normalised_mean():
    f = stored_features([([1.0, 0.0], "like"), ([0.0, 1.0], "like")])
    assert f["id-close"][1] == pytest.approx(2 ** -0.5)


def test_feedback_changes_what_gets_picked():
    papers = [candidate("close"), candidate("medium"), candidate("far")]
    assert rank.rank(RankConn(papers), n=1) == ["id-close"]
    assert rank.rank(RankConn(papers, [([0.0, 1.0], "like")]), n=1) == ["id-medium"]


def test_rank_with_no_candidates_returns_nothing():
    assert rank.rank(RankConn([])) == []


def test_unpick_clears_everything_rank_set():
    conn = RankConn([])
    rank.unpick(conn, ["id-close"])
    assert "embedding = null, features = null, selected_at = null" in conn.last
    assert conn.updates == [(["id-close"],)]
