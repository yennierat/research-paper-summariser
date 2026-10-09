import logging
from langfuse import observe, get_client
from src.db import connect
from src.sources import Paper, fetch_arxiv, fetch_hf

log = logging.getLogger(__name__)

RETENTION_DAYS = 7


def save(conn, papers: list[Paper]) -> None:
    conn.cursor().executemany(
        """insert into papers (arxiv_id, title, abstract, authors, categories, published, hf_upvotes)
           values (%s, %s, %s, %s, %s, %s, %s)
           on conflict (arxiv_id) do update
           set hf_upvotes = coalesce(excluded.hf_upvotes, papers.hf_upvotes)""",
        [(p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.published, p.hf_upvotes)
         for p in papers],
    )


def prune(conn) -> int:
    """Unpicked papers stay rankable for RETENTION_DAYS, then go. Picked papers are kept for training."""
    return conn.execute(
        "delete from papers where selected_at is null and published < now() - make_interval(days => %s)",
        (RETENTION_DAYS,),
    ).rowcount


@observe(name="ingest")
def ingest() -> None:
    papers = fetch_arxiv()
    log.info("fetched %d papers from arXiv", len(papers))
    try:
        hf = fetch_hf()
        log.info("fetched %d papers from HF Daily Papers", len(hf))
        papers += hf
    except Exception:
        log.exception("HF Daily Papers failed, continuing with arXiv only")
    with connect() as conn:
        save(conn, papers)
        deleted = prune(conn)
    log.info("saved %d papers, pruned %d unpicked papers older than %d days",
             len(papers), deleted, RETENTION_DAYS)
    get_client().update_current_span(output={"fetched": len(papers), "pruned": deleted})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
