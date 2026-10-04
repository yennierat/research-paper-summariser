from datetime import datetime, timezone
from src import sources
from tests.fakes import FakeResponse


def arxiv_feed(arxiv_url: str) -> bytes:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>{arxiv_url}</id>
    <title>One Basis to Animate
      Them All</title>
    <summary>  Line one
    line two. </summary>
    <published>2026-10-01T17:59:58Z</published>
    <category term="cs.CV"/>
    <category term="cs.LG"/>
    <author><name>Ramazan Fazylov</name></author>
    <author><name>Ivan Laptev</name></author>
  </entry>
</feed>""".encode()


def test_fetch_arxiv_parses_entry(monkeypatch):
    monkeypatch.setattr(sources.requests, "get",
                        lambda *a, **k: FakeResponse(content=arxiv_feed("http://arxiv.org/abs/2610.02207v2")))
    [p] = sources.fetch_arxiv()
    assert p.arxiv_id == "2610.02207"
    assert p.title == "One Basis to Animate Them All"
    assert p.abstract == "Line one line two."
    assert p.authors == ["Ramazan Fazylov", "Ivan Laptev"]
    assert p.categories == ["cs.CV", "cs.LG"]
    assert p.published == datetime(2026, 10, 1, 17, 59, 58, tzinfo=timezone.utc)
    assert p.hf_upvotes is None


def test_fetch_arxiv_handles_old_style_ids(monkeypatch):
    monkeypatch.setattr(sources.requests, "get",
                        lambda *a, **k: FakeResponse(content=arxiv_feed("http://arxiv.org/abs/hep-th/9901001v1")))
    assert sources.fetch_arxiv()[0].arxiv_id == "hep-th/9901001"


def test_fetch_arxiv_passes_max_results(monkeypatch):
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return FakeResponse(content=arxiv_feed("http://arxiv.org/abs/2610.02207v1"))

    monkeypatch.setattr(sources.requests, "get", fake_get)
    sources.fetch_arxiv(max_results=1)
    assert seen["max_results"] == 1
    assert "cat:cs.CR" in seen["search_query"]


def test_fetch_hf_parses_paper(monkeypatch):
    data = [{"paper": {"id": "2609.22753", "title": " A   title ", "summary": "Some\n summary",
                       "authors": [{"name": "Delong Li"}], "publishedAt": "2026-09-26T00:00:00.000Z",
                       "upvotes": 5}}]
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(data=data))
    [p] = sources.fetch_hf()
    assert (p.arxiv_id, p.title, p.abstract, p.authors) == ("2609.22753", "A title", "Some summary", ["Delong Li"])
    assert p.hf_upvotes == 5
    assert p.categories == []
    assert p.published.tzinfo is not None
