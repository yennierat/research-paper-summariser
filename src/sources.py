import re
import time
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
import requests

log = logging.getLogger(__name__)

ARXIV_CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.CR"]
ARXIV_MAX_RESULTS = 1000
ARXIV_TRIES = 5  # waits 5, 10, 20, 40s between tries
BACKOFF_BASE = 5  # arXiv asks for at least 3s between requests
MAX_RETRY_AFTER = 120
RETRY_STATUSES = {429, 500, 502, 503, 504}
NS = {"a": "http://www.w3.org/2005/Atom"}

FULLTEXT_TRIES = 3
FULLTEXT_MAX_CHARS = 100_000  # ~25k tokens; 4 of 5 sampled papers (40k-105k chars) fit whole
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SKIP_TAGS = {"head", "script", "style", "nav", "header", "footer", "button", "form", "svg"}
SKIP_CLASSES = {"ltx_bibliography", "ltx_TOC", "ltx_toclist", "ltx_appendix", "ltx_authors",
                "ltx_eqn_table"}  # display equations are tables of inline math, so display="block" misses them
BLOCK_TAGS = {"p", "div", "section", "article", "li", "tr", "table", "figcaption", "caption",
              "h1", "h2", "h3", "h4", "h5", "h6"}


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


class _PaperText(HTMLParser):
    """Plain text of an arXiv (LaTeXML) HTML paper. Inline math becomes its LaTeX alttext, so numbers
    written in math mode ($70.24\\%$) survive; display equations, references, TOC and appendix are dropped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines, self.line = [], []
        self.stack = []  # (tag, whether it started a skipped subtree, whether it is the <article>)
        self.skipping = 0
        self.in_article = 0  # text outside <article> is arXiv's page chrome (banners, license, menus)

    def _reading(self) -> bool:
        return self.in_article > 0 and not self.skipping

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        a = dict(attrs)
        skip = False
        if self._reading():
            if tag == "math":
                if a.get("display") != "block":
                    self.line.append(f" {a.get('alttext') or ''} ")
                skip = True  # the MathML children would repeat the alttext
            elif tag in SKIP_TAGS or SKIP_CLASSES & set((a.get("class") or "").split()):
                skip = True
            elif tag in ("td", "th") and "".join(self.line).strip():
                self.line.append(" | ")
        article = tag == "article"
        self.stack.append((tag, skip, article))
        self.skipping += skip
        self.in_article += article

    def handle_endtag(self, tag):
        if tag in VOID_TAGS or all(t != tag for t, _, _ in self.stack):
            return
        reading = self._reading()
        while self.stack:  # also closes tags the page left open
            t, skip, article = self.stack.pop()
            self.skipping -= skip
            self.in_article -= article
            if t == tag:
                break
        if reading and tag in BLOCK_TAGS:
            self._end_line()

    def handle_data(self, data):
        if self._reading():
            self.line.append(data)

    def _end_line(self):
        if line := _clean("".join(self.line)):
            self.lines.append(line)
        self.line = []

    def text(self) -> str:
        self._end_line()
        return "\n".join(self.lines)


def fetch_fulltext(arxiv_id: str) -> str | None:
    """Text of the paper's arXiv HTML version, or None when it has none (conversion lags for new papers)."""
    try:
        r = _get(f"https://arxiv.org/html/{arxiv_id}", FULLTEXT_TRIES, timeout=30)
    except requests.RequestException as e:
        log.warning("no full text for %s (%s), using the abstract", arxiv_id, type(e).__name__)
        return None
    if "ltx_document" not in r.text:
        log.warning("arXiv HTML for %s is not a converted paper, using the abstract", arxiv_id)
        return None
    parser = _PaperText()
    parser.feed(r.text)
    parser.close()
    return parser.text() or None


def fit(text: str, max_chars: int = FULLTEXT_MAX_CHARS) -> str:
    """Trim to max_chars, keeping the start (intro) and the end (conclusion)."""
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    return f"{text[:head]}\n[...]\n{text[-(max_chars - head):]}"


def paper_text(arxiv_id: str, abstract: str) -> tuple[str, dict]:
    """What to summarize: the fitted full text when arXiv has it, else the abstract. Plus where it came from."""
    full = fetch_fulltext(arxiv_id)
    if full is None:
        return abstract, {"source": "abstract", "chars": len(abstract), "truncated": False}
    return fit(full), {"source": "fulltext", "chars": len(full), "truncated": len(full) > FULLTEXT_MAX_CHARS}


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
