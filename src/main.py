import logging
from src.db import connect
from src.ingest import ingest
from src.poller import poll
from src.rank import rank, unpick
from src.summarize import Summarizer, MODEL, missing_numbers, describe_error
from src.telegram import Telegram, format_card, feedback_buttons

log = logging.getLogger(__name__)


def alert(text: str) -> None:
    """Best-effort error notification; a broken Telegram mustn't hide the original error."""
    try:
        Telegram().send_text(f"[ALERT] {text}", parse_mode=None)
    except Exception:
        log.exception("could not send alert")


def digest() -> None:
    """Fetch, rank and send today's papers. Uses the hand-set weights until retrain promotes a model."""
    try:
        poll()  # with the evening poller, keeps taps well inside Telegram's 24h retention
    except Exception as e:
        log.exception("feedback poll failed, continuing")
        alert(f"Feedback poll failed, sending digest anyway: {describe_error(e)}")
    ingest()
    with connect() as conn:
        ids = rank(conn)
        rows = conn.execute(
            "select arxiv_id, title, abstract from papers where arxiv_id = any(%s)", (ids,)
        ).fetchall()
    tg = Telegram()
    if not ids:
        tg.send_text("No new papers today.", parse_mode=None)
        return

    papers = {r[0]: r for r in rows}
    summarizer = Summarizer()
    failures = {}
    for arxiv_id in ids:
        _, title, abstract = papers[arxiv_id]
        try:
            s = summarizer.summarize(title, abstract)
            missing_numbers(s, abstract)
            text = "\n\n".join([s.intro.short, s.results.short, s.discussion.short])
            tg.send_text(format_card(arxiv_id, title, text, MODEL),
                         reply_markup=feedback_buttons(arxiv_id))
        except Exception as e:
            log.exception("failed to summarize or send %s, skipping", arxiv_id)
            failures[arxiv_id] = describe_error(e)
    log.info("sent %d of %d papers", len(ids) - len(failures), len(ids))
    if failures:
        with connect() as conn:
            unpick(conn, list(failures))
        report = (f"{len(failures)} of {len(ids)} papers failed, returned to the pool\n"
                  + "\n".join(f"{i}: {d}" for i, d in failures.items()))
        if len(failures) == len(ids):
            raise RuntimeError(report)
        alert(f"Digest: {report}")


def main() -> None:
    try:
        digest()
    except Exception as e:
        alert(f"Digest failed: {describe_error(e)}")
        raise


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    main()
