from datetime import datetime, timezone
import pytest
import requests
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


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(sources.time, "sleep", waits.append)
    return waits


def responses(*items):
    """Fake requests.get that returns (or raises) each item in turn."""
    calls = []

    def get(*a, **k):
        item = items[len(calls)]
        calls.append(1)
        if isinstance(item, Exception):
            raise item
        return item

    return get, calls


def ok_feed():
    return FakeResponse(content=arxiv_feed("http://arxiv.org/abs/2610.02207v1"))


def test_fetch_arxiv_retries_with_exponential_backoff(monkeypatch, sleeps):
    get, calls = responses(FakeResponse(status_code=503), requests.ConnectionError("reset"),
                           requests.ReadTimeout("slow"), ok_feed())
    monkeypatch.setattr(sources.requests, "get", get)
    assert sources.fetch_arxiv()[0].arxiv_id == "2610.02207"
    assert sleeps == [5, 10, 20]


def test_fetch_arxiv_honours_retry_after(monkeypatch, sleeps):
    get, _ = responses(FakeResponse(status_code=429, headers={"Retry-After": "30"}),
                       FakeResponse(status_code=503, headers={"Retry-After": "9999"}), ok_feed())
    monkeypatch.setattr(sources.requests, "get", get)
    sources.fetch_arxiv()
    assert sleeps == [30, sources.MAX_RETRY_AFTER]


def test_fetch_arxiv_gives_up_after_max_tries(monkeypatch, sleeps):
    get, calls = responses(*[FakeResponse(status_code=503)] * sources.ARXIV_TRIES)
    monkeypatch.setattr(sources.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        sources.fetch_arxiv()
    assert len(calls) == sources.ARXIV_TRIES
    assert sleeps == [5, 10, 20, 40]


def test_fetch_arxiv_reraises_last_connection_error(monkeypatch, sleeps):
    get, calls = responses(*[requests.ConnectionError("reset")] * sources.ARXIV_TRIES)
    monkeypatch.setattr(sources.requests, "get", get)
    with pytest.raises(requests.ConnectionError):
        sources.fetch_arxiv()
    assert len(calls) == sources.ARXIV_TRIES


def test_fetch_arxiv_does_not_retry_client_errors(monkeypatch, sleeps):
    get, calls = responses(FakeResponse(status_code=400))
    monkeypatch.setattr(sources.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        sources.fetch_arxiv()
    assert len(calls) == 1
    assert sleeps == []


PAPER_HTML = r"""<html><head><title>page title</title><script>var x = 1;</script></head><body>
<header><span class="ds-announcement-text">arXiv is now an independent nonprofit!</span></header>
<div class="ltx_page_header">License: arXiv.org perpetual license</div>
<article class="ltx_document">
<h1 class="ltx_title">A Paper</h1>
<div class="ltx_authors">Ann Author, ann@example.com</div>
<nav class="ltx_TOC"><ol class="ltx_toclist"><li>1 Introduction</li></ol></nav>
<div class="ltx_abstract"><h6>Abstract</h6><p class="ltx_p">We   study
things.</p></div>
<section class="ltx_section"><h2>1 Results</h2>
<p>Accuracy reached <math alttext="70.24\%" display="inline"><semantics><mn>70.24</mn><mo>%</mo></semantics></math> on the test set.<br>
A display equation follows.</p>
<table class="ltx_equation ltx_eqn_table"><tr><td><math alttext="\displaystyle E=mc^{2}" display="inline"><mi>E</mi></math></td></tr></table>
<math alttext="x^2" display="block"><mi>x</mi></math>
<figure><table class="ltx_tabular"><tr><th>Method</th><th>Score</th></tr><tr><td>Ours</td><td>0.91</td></tr></table>
<figcaption>Table 1: Main results.</figcaption></figure>
</section>
<section class="ltx_appendix"><h2>Appendix A</h2><p>Extra proofs.</p></section>
<section class="ltx_bibliography"><h2>References</h2><ul><li class="ltx_bibitem">[1] Someone. 2020.</li></ul></section>
</article>
<footer>About arXiv</footer>
</body></html>"""


def test_fetch_fulltext_keeps_paper_text_and_drops_the_rest(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(text=PAPER_HTML))
    assert sources.fetch_fulltext("2610.00001").splitlines() == [
        "A Paper",
        "Abstract",
        "We study things.",
        "1 Results",
        r"Accuracy reached 70.24\% on the test set. A display equation follows.",
        "Method | Score",
        "Ours | 0.91",
        "Table 1: Main results.",
    ]


def test_fetch_fulltext_returns_none_without_an_html_version(monkeypatch, sleeps):
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(status_code=404))
    assert sources.fetch_fulltext("2610.00001") is None


def test_fetch_fulltext_returns_none_when_arxiv_is_unreachable(monkeypatch, sleeps):
    get, calls = responses(*[requests.ConnectionError("reset")] * sources.FULLTEXT_TRIES)
    monkeypatch.setattr(sources.requests, "get", get)
    assert sources.fetch_fulltext("2610.00001") is None
    assert len(calls) == sources.FULLTEXT_TRIES


def test_fetch_fulltext_returns_none_for_a_page_that_is_not_a_paper(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(text="<html>No HTML for this paper</html>"))
    assert sources.fetch_fulltext("2610.00001") is None


def test_fit_keeps_short_text_and_cuts_the_middle_of_long_text():
    assert sources.fit("short", max_chars=10) == "short"
    assert sources.fit("a" * 15 + "b" * 10 + "c" * 5, max_chars=20) == "a" * 15 + "\n[...]\n" + "c" * 5


def test_paper_text_prefers_full_text(monkeypatch):
    long_text = "x" * (sources.FULLTEXT_MAX_CHARS + 10)
    monkeypatch.setattr(sources, "fetch_fulltext", lambda arxiv_id: long_text)
    text, source = sources.paper_text("2610.00001", "the abstract")
    assert text == sources.fit(long_text)
    assert source == {"source": "fulltext", "chars": len(long_text), "truncated": True}


def test_paper_text_marks_whole_papers_as_not_truncated(monkeypatch):
    monkeypatch.setattr(sources, "fetch_fulltext", lambda arxiv_id: "the whole paper")
    assert sources.paper_text("2610.00001", "the abstract") == (
        "the whole paper", {"source": "fulltext", "chars": 15, "truncated": False})


def test_paper_text_falls_back_to_the_abstract(monkeypatch):
    monkeypatch.setattr(sources, "fetch_fulltext", lambda arxiv_id: None)
    assert sources.paper_text("2610.00001", "the abstract") == (
        "the abstract", {"source": "abstract", "chars": 12, "truncated": False})
