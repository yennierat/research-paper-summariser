import os
import html
import logging
import random
import time
import requests

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096
MAX_TRIES = 3
LABELS = {"like": "Like", "dislike": "Not for me", "save": "Save"}


class Telegram:
    """Telegram delivery of Research Articles summary"""
    def __init__(self):
        self.token = os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat_id = os.environ["TELEGRAM_CHAT_ID"]

    def send_text(self, summary: str, parse_mode: str | None = "HTML",
                  reply_markup: dict | None = None) -> None:
        payload = {"chat_id": self.chat_id, "text": summary[:TELEGRAM_LIMIT],
                   "link_preview_options": {"is_disabled": True}}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup
        r = self._request("post", "sendMessage", json=payload)
        if not r.ok:
            raise RuntimeError(f"Telegram sendMessage failed ({r.status_code}): {r.text}")
        log.info("sent Telegram message (%d chars)", len(summary))

    def get_updates(self, offset: int | None) -> list[dict]:
        """Fetching with offset confirms every earlier update, so Telegram won't resend it."""
        r = self._request("get", "getUpdates",
                          params={"offset": offset, "allowed_updates": '["callback_query"]'})
        if not r.ok:
            raise RuntimeError(f"Telegram getUpdates failed ({r.status_code}): {r.text}")
        return r.json()["result"]

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        for attempt in range(MAX_TRIES):
            try:
                return requests.request(method, f"https://api.telegram.org/bot{self.token}/{endpoint}",
                                        timeout=(10, 20), **kwargs)
            except requests.ConnectionError as e:
                if attempt == MAX_TRIES - 1:
                    raise
                wait = 2 ** attempt + random.uniform(0, 1)
                log.warning("Telegram %s connection failed (%s), retrying in %.1fs",
                            endpoint, type(e).__name__, wait)
                time.sleep(wait)


def feedback_buttons(arxiv_id: str) -> dict:
    return {"inline_keyboard": [[
        {"text": text, "callback_data": f"{label}:{arxiv_id}"} for label, text in LABELS.items()
    ]]}


def format_card(arxiv_id: str, title: str, summary: str, model: str) -> str:
    """Build a paper card as Telegram HTML. Paper text is escaped so < > & can't break it."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    return (f"<b>{html.escape(title)}</b>\n"
            f'<a href="{url}">arXiv:{arxiv_id}</a>\n\n'
            f"{html.escape(summary)}\n\n"
            f"<i>Summarized by {html.escape(model)}</i>")
