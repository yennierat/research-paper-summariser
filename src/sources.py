import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
import requests

ARXIV_CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.CR"]
ARXIV_MAX_RESULTS = 1000
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


def fetch_arxiv(max_results: int = ARXIV_MAX_RESULTS) -> list[Paper]:
    r = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        },
        timeout=60,
    )
    r.raise_for_status()
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
