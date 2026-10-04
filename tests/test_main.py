import pytest

pytest.importorskip("sentence_transformers")
from src import main  # noqa: E402
from src.summarize import Section, Summary  # noqa: E402
from tests.fakes import FakeConn  # noqa: E402

IDS = [f"2610.0000{i}" for i in range(5)]
ROWS = [(i, f"Title {i}", f"Abstract {i}") for i in reversed(IDS)]


class FakeTelegram:
    def __init__(self):
        self.messages = []

    def send_text(self, text, parse_mode="HTML", reply_markup=None):
        self.messages.append((text, reply_markup))


class FakeSummarizer:
    fail_on = None

    def summarize(self, title, text):
        if title == self.fail_on:
            raise RuntimeError("OpenRouter down")
        return Summary(**{k: Section(short=f"{k} of {title}", long="") for k in ("intro", "results", "discussion")})


@pytest.fixture
def fake(monkeypatch):
    """Installs fakes for one digest run; returns the Telegram and the DB connection it uses."""
    def fake(ids, rows):
        tg, conn = FakeTelegram(), FakeConn(rows)
        monkeypatch.setattr(main, "poll", lambda: None)
        monkeypatch.setattr(main, "ingest", lambda: None)
        monkeypatch.setattr(main, "connect", lambda: conn)
        monkeypatch.setattr(main, "rank", lambda conn: ids)
        monkeypatch.setattr(main, "Summarizer", FakeSummarizer)
        monkeypatch.setattr(main, "Telegram", lambda: tg)
        return tg, conn

    return fake


@pytest.fixture
def run(fake):
    def run(ids, rows):
        tg, _ = fake(ids, rows)
        main.digest()
        return tg.messages

    return run


def test_sends_one_card_with_buttons_per_pick_in_rank_order(fake):
    tg, conn = fake(IDS, ROWS)
    main.digest()
    assert len(tg.messages) == 5
    for arxiv_id, (text, buttons) in zip(IDS, tg.messages):
        assert f"arXiv:{arxiv_id}" in text
        assert buttons["inline_keyboard"][0][0]["callback_data"] == f"like:{arxiv_id}"
    assert "intro of Title 2610.00000" in tg.messages[0][0]
    assert not conn.statements("selected_at = null")


def test_failed_paper_is_skipped_unpicked_and_reported(fake, monkeypatch):
    monkeypatch.setattr(FakeSummarizer, "fail_on", "Title 2610.00002")
    tg, conn = fake(IDS, ROWS)
    main.digest()
    *cards, (report, _) = tg.messages
    assert len(cards) == 4
    assert not any("2610.00002" in text for text, _ in cards)
    assert [p for _, p in conn.statements("selected_at = null")] == [(["2610.00002"],)]
    assert report.startswith("[ALERT] Digest: 1 of 5 papers failed, returned to the pool")
    assert "2610.00002: RuntimeError: OpenRouter down" in report


def test_alerts_and_fails_when_nothing_could_be_sent(fake, monkeypatch):
    monkeypatch.setattr(FakeSummarizer, "fail_on", "Title 2610.00000")
    tg, conn = fake(IDS[:1], ROWS)
    with pytest.raises(RuntimeError):
        main.main()
    assert [p for _, p in conn.statements("selected_at = null")] == [(["2610.00000"],)]
    [(text, _)] = tg.messages
    assert text.startswith("[ALERT] Digest failed: RuntimeError: 1 of 1 papers failed")
    assert "2610.00000: RuntimeError: OpenRouter down" in text


def test_polls_feedback_before_ranking(fake, monkeypatch):
    calls = []
    fake(IDS, ROWS)
    monkeypatch.setattr(main, "poll", lambda: calls.append("poll"))
    monkeypatch.setattr(main, "rank", lambda conn: calls.append("rank") or IDS)
    main.digest()
    assert calls == ["poll", "rank"]


def test_failed_poll_alerts_but_still_sends_digest(fake, monkeypatch):
    tg, _ = fake(IDS, ROWS)

    def broken():
        raise RuntimeError("Telegram getUpdates failed (502)")
    monkeypatch.setattr(main, "poll", broken)
    main.digest()
    (first, _), *cards = tg.messages
    assert first.startswith("[ALERT] Feedback poll failed, sending digest anyway")
    assert len(cards) == 5


def test_alert_never_raises_when_telegram_is_broken(monkeypatch):
    def broken():
        raise KeyError("TELEGRAM_BOT_TOKEN")
    monkeypatch.setattr(main, "Telegram", broken)
    main.alert("something broke")


def test_no_candidates_sends_a_note(run):
    assert run([], []) == [("No new papers today.", None)]
