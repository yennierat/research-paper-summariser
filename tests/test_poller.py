from src import poller
from tests.fakes import FakeConn


class FakeTelegram:
    def __init__(self, batches):
        self.batches, self.offsets = batches, []

    def get_updates(self, offset):
        self.offsets.append(offset)
        return self.batches.pop(0) if self.batches else []


def test_poll_saves_valid_taps_and_confirms_all(monkeypatch):
    tg = FakeTelegram([
        [
            {"update_id": 10, "callback_query": {"data": "dislike:2610.02207"}},
            {"update_id": 11, "message": {"text": "hi"}},
            {"update_id": 12, "callback_query": {"data": "bogus:2610.00001"}},
        ],
        [{"update_id": 13, "callback_query": {"data": "save:2609.22753"}}],
    ])
    conn = FakeConn()
    monkeypatch.setattr(poller, "Telegram", lambda: tg)
    monkeypatch.setattr(poller, "connect", lambda: conn)
    poller.poll()
    assert [p for _, p in conn.statements("insert into feedback")] == [
        (10, "2610.02207", "dislike"),
        (13, "2609.22753", "save"),
    ]
    assert tg.offsets == [None, 13, 14]


def test_poll_with_no_updates_does_nothing(monkeypatch):
    tg, conn = FakeTelegram([]), FakeConn()
    monkeypatch.setattr(poller, "Telegram", lambda: tg)
    monkeypatch.setattr(poller, "connect", lambda: conn)
    poller.poll()
    assert not conn.statements("insert")
    assert tg.offsets == [None]
