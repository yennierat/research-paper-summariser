import pytest
import requests
from src import telegram
from src.telegram import Telegram, format_card, feedback_buttons, LABELS
from tests.fakes import FakeResponse


@pytest.fixture
def tg(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)
    return Telegram()


def test_format_card_escapes_paper_text():
    card = format_card("2610.02207", "A <b> & c", "p < 0.05", "openai/gpt-6-luna")
    assert "<b>A &lt;b&gt; &amp; c</b>" in card
    assert "p &lt; 0.05" in card
    assert '<a href="https://arxiv.org/abs/2610.02207">arXiv:2610.02207</a>' in card
    assert card.endswith("<i>Summarized by openai/gpt-6-luna</i>")


def test_feedback_buttons_encode_label_and_paper():
    buttons = feedback_buttons("2610.02207")["inline_keyboard"][0]
    assert [b["callback_data"] for b in buttons] == [f"{label}:2610.02207" for label in LABELS]
    assert all(len(b["callback_data"].encode()) <= 64 for b in buttons)


def test_send_text_payload(tg, monkeypatch):
    sent = {}

    def fake_request(method, url, timeout, json):
        sent.update(method=method, url=url, json=json)
        return FakeResponse()

    monkeypatch.setattr(telegram.requests, "request", fake_request)
    tg.send_text("hello", reply_markup=feedback_buttons("1"))
    assert sent["method"] == "post"
    assert sent["url"].endswith("/bottest-token/sendMessage")
    assert sent["json"]["chat_id"] == "42"
    assert sent["json"]["parse_mode"] == "HTML"
    assert "inline_keyboard" in sent["json"]["reply_markup"]


def test_send_text_plain_text_has_no_parse_mode(tg, monkeypatch):
    sent = {}
    monkeypatch.setattr(telegram.requests, "request", lambda m, u, timeout, json: sent.update(json) or FakeResponse())
    tg.send_text("hello", parse_mode=None)
    assert "parse_mode" not in sent


def test_retries_connection_errors_then_succeeds(tg, monkeypatch):
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError("reset")
        return FakeResponse()

    monkeypatch.setattr(telegram.requests, "request", flaky)
    tg.send_text("hello")
    assert len(calls) == 3


def test_gives_up_after_three_tries(tg, monkeypatch):
    calls = []

    def down(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("reset")

    monkeypatch.setattr(telegram.requests, "request", down)
    with pytest.raises(requests.ConnectionError):
        tg.send_text("hello")
    assert len(calls) == telegram.MAX_TRIES


def test_telegram_error_is_raised_without_retry(tg, monkeypatch):
    calls = []

    def bad(*a, **k):
        calls.append(1)
        return FakeResponse(ok=False, status_code=400, text="can't parse entities")

    monkeypatch.setattr(telegram.requests, "request", bad)
    with pytest.raises(RuntimeError, match="400"):
        tg.send_text("<b>")
    assert len(calls) == 1


def test_get_updates_returns_result(tg, monkeypatch):
    monkeypatch.setattr(telegram.requests, "request",
                        lambda *a, **k: FakeResponse(data={"ok": True, "result": [{"update_id": 1}]}))
    assert tg.get_updates(None) == [{"update_id": 1}]
