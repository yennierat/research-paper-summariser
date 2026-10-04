from datetime import datetime, timezone
import pytest
from src import ingest
from src.sources import Paper
from tests.fakes import FakeConn

PAPER = Paper("2610.02207", "Title", "Abstract", ["A"], datetime(2026, 10, 1, tzinfo=timezone.utc), ["cs.LG"])
HF_PAPER = Paper("2609.22753", "HF", "Abstract", ["B"], datetime(2026, 9, 26, tzinfo=timezone.utc), hf_upvotes=5)


@pytest.fixture
def conn(monkeypatch):
    c = FakeConn()
    monkeypatch.setattr(ingest, "connect", lambda: c)
    monkeypatch.setattr(ingest, "fetch_arxiv", lambda: [PAPER])
    return c


def saved_ids(conn):
    [(_, params)] = conn.statements("insert into papers")
    return [p[0] for p in params]


def test_ingest_saves_both_sources(monkeypatch, conn):
    monkeypatch.setattr(ingest, "fetch_hf", lambda: [HF_PAPER])
    ingest.ingest()
    assert saved_ids(conn) == ["2610.02207", "2609.22753"]


def test_ingest_continues_when_hf_is_down(monkeypatch, conn):
    def down():
        raise ConnectionError("HF down")

    monkeypatch.setattr(ingest, "fetch_hf", down)
    ingest.ingest()
    assert saved_ids(conn) == ["2610.02207"]


def test_ingest_prunes_only_unpicked_old_papers(monkeypatch, conn):
    monkeypatch.setattr(ingest, "fetch_hf", lambda: [])
    ingest.ingest()
    [(sql, params)] = conn.statements("delete from papers")
    assert "selected_at is null" in sql
    assert params == (ingest.RETENTION_DAYS,)
