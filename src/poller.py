import logging
from src.db import connect
from src.telegram import Telegram, LABELS

log = logging.getLogger(__name__)


def poll() -> None:
    tg = Telegram()
    offset = None
    saved = 0
    with connect() as conn:
        while updates := tg.get_updates(offset):
            for u in updates:
                label, _, arxiv_id = u.get("callback_query", {}).get("data", "").partition(":")
                if label in LABELS and arxiv_id:
                    conn.execute(
                        "insert into feedback (update_id, arxiv_id, label) values (%s, %s, %s)"
                        " on conflict (update_id) do nothing",
                        (u["update_id"], arxiv_id, label),
                    )
                    saved += 1
            conn.commit()
            offset = updates[-1]["update_id"] + 1
    log.info("saved %d feedback taps", saved)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    poll()
