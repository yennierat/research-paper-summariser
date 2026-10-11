import httpx2
import psycopg
import requests
from langfuse.openai import openai
from src.summarize import Summary, missing_numbers, describe_error

REQUEST = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def summary(results: str) -> Summary:
    return Summary(intro="x", results=results, discussion="x")


def test_numbers_from_source_pass():
    s = summary("Accuracy rose to 92.5% on 1,000 images.")
    assert missing_numbers(s, "We reach 92.5 accuracy on 1000 images.") == []


def test_invented_numbers_are_flagged():
    s = summary("3x faster, trained on 12 GPUs.")
    assert missing_numbers(s, "Much faster, trained on 8 GPUs.") == ["12", "3"]


def test_only_results_section_is_checked():
    s = Summary(intro="99 problems", results="no numbers", discussion="")
    assert missing_numbers(s, "") == []


def test_rate_limit_is_named_as_429():
    e = openai.RateLimitError("Rate limit exceeded", response=httpx2.Response(429, request=REQUEST), body=None)
    assert "rate limit (HTTP 429)" in describe_error(e)


def test_other_http_errors_show_the_status():
    e = openai.InternalServerError("upstream error", response=httpx2.Response(502, request=REQUEST), body=None)
    assert describe_error(e) == "OpenRouter HTTP 502: upstream error"


def test_connection_errors_are_described():
    assert describe_error(openai.APIConnectionError(request=REQUEST)) == "could not reach OpenRouter"


def test_request_errors_name_the_host_but_not_the_url():
    req = requests.Request("POST", "https://api.telegram.org/botSECRET-TOKEN/sendMessage")
    e = requests.ConnectionError("HTTPSConnectionPool ... url: /botSECRET-TOKEN/sendMessage", request=req)
    assert describe_error(e) == "ConnectionError reaching api.telegram.org"


def test_database_errors_hide_the_message():
    e = psycopg.ProgrammingError('invalid connection option ""postgresql://user:SECRET-PW@ep-host.neon.tech/db')
    assert describe_error(e) == "database error (ProgrammingError)"


def test_unknown_errors_keep_type_and_message():
    assert describe_error(ValueError("bad row")) == "ValueError: bad row"
