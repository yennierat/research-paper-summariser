import os
import re
import time
import logging
from urllib.parse import urlparse
import psycopg
import requests
from langfuse.openai import OpenAI, openai
from pydantic import BaseModel

log = logging.getLogger(__name__)

MODEL = "openai/gpt-6-luna"

PROMPT = """Summarize this research paper for a busy reader.
For each of intro, results and discussion give:
- short: 1-2 sentences
- long: one paragraph
Only use facts and numbers that appear in the text. Do not include links."""


class Section(BaseModel):
    short: str
    long: str


class Summary(BaseModel):
    intro: Section
    results: Section
    discussion: Section


class Summarizer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def summarize(self, title: str, text: str) -> Summary:
        start = time.monotonic()
        r = self.client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Title: {title}\n\n{text}"},
            ],
            response_format=Summary,
        )
        log.info("summarized %r with %s in %.1fs (%d in / %d out tokens)",
                 title, r.model, time.monotonic() - start,
                 r.usage.prompt_tokens, r.usage.completion_tokens)
        return r.choices[0].message.parsed


def missing_numbers(summary: Summary, source: str) -> list[str]:
    """Numbers in the results summary that don't appear in the source text."""
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s.replace(",", "")))
    missing = sorted(nums(summary.results.short + " " + summary.results.long) - nums(source))
    if missing:
        log.warning("results summary has numbers not in source: %s", missing)
    return missing


def describe_error(e: Exception) -> str:
    """One readable line for an alert. The client has already retried 429s and 5xx by the time we see them."""
    if isinstance(e, openai.RateLimitError):
        return "OpenRouter rate limit (HTTP 429), still failing after retries"
    if isinstance(e, openai.APIStatusError):
        return f"OpenRouter HTTP {e.status_code}: {e.message[:200]}"
    if isinstance(e, openai.APIConnectionError):
        return "could not reach OpenRouter"
    if isinstance(e, requests.RequestException):
        host = urlparse(e.request.url).hostname if e.request is not None else "remote host"
        return f"{type(e).__name__} reaching {host}"
    if isinstance(e, psycopg.Error):
        return f"database error ({type(e).__name__})"
    return f"{type(e).__name__}: {e}"
