import re
import time
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
import requests

log = logging.getLogger(__name__)

ARXIV_CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.CR"]
ARXIV_MAX_RESULTS = 1000
ARXIV_TRIES = 5  # waits 5, 10, 20, 40s between tries
BACKOFF_BASE = 5  # arXiv asks for at least 3s between requests
MAX_RETRY_AFTER = 120
RETRY_STATUSES = {429, 500, 502, 503, 504}
NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    categories: list[str] = field(default_factory=list)
    hf_upvotes: int | None = None


def _clean(text: str) -> str:
    return " ".join(text.split())


def _get(url: str, tries: int, **kwargs) -> requests.Response:
    """GET with exponential backoff on connection errors, timeouts, 429 and 5xx. Other errors raise at once."""
    for attempt in range(tries):
        last = attempt == tries - 1
        wait = BACKOFF_BASE * 2 ** attempt
        try:
            r = requests.get(url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            if last:
                raise
            reason = type(e).__name__
        else:
            if r.status_code not in RETRY_STATUSES or last:
                r.raise_for_status()
                return r
            reason = f"HTTP {r.status_code}"
            retry_after = r.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, min(int(retry_after), MAX_RETRY_AFTER))
        log.warning("GET %s failed (%s), retrying in %ds (try %d/%d)",
                    url, reason, wait, attempt + 1, tries)
        time.sleep(wait)


def fetch_arxiv(max_results: int = ARXIV_MAX_RESULTS) -> list[Paper]:
    r = _get(
        "https://export.arxiv.org/api/query",
        ARXIV_TRIES,
        params={
            "search_query": " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        },
        timeout=60,
    )
    papers = []
    for e in ET.fromstring(r.content).findall("a:entry", NS):
        papers.append(Paper(
            arxiv_id=re.sub(r"v\d+$", "", e.findtext("a:id", namespaces=NS).split("/abs/")[-1]),
            title=_clean(e.findtext("a:title", namespaces=NS)),
            abstract=_clean(e.findtext("a:summary", namespaces=NS)),
            authors=[a.findtext("a:name", namespaces=NS) for a in e.findall("a:author", NS)],
            published=datetime.fromisoformat(e.findtext("a:published", namespaces=NS)),
            categories=[c.get("term") for c in e.findall("a:category", NS)],
        ))
    return papers


def fetch_hf() -> list[Paper]:
    r = requests.get("https://huggingface.co/api/daily_papers", timeout=30)
    r.raise_for_status()
    return [
        Paper(
            arxiv_id=p["id"],
            title=_clean(p["title"]),
            abstract=_clean(p["summary"]),
            authors=[a["name"] for a in p["authors"]],
            published=datetime.fromisoformat(p["publishedAt"]),
            hf_upvotes=p["upvotes"],
        )
        for p in (item["paper"] for item in r.json())
    ]
