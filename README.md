# Research Paper Recommender

Every day it pulls new papers from arXiv and Hugging Face Daily Papers, ranks them against your interests and past feedback, summarizes the top 5 with an LLM, and sends each one to Telegram with **Like / Not for me / Save** buttons. Your taps train the ranker.

Three GitHub Actions jobs, no server. They share state only through Postgres (Neon).

| Job | Runs | Entry point |
|---|---|---|
| digest | every day | `python -m src.main` |
| poller | every day | `python -m src.poller` |
| retrain | every week | `python -m src.retrain` |

GitHub often starts scheduled jobs hours late. Each job can also be run by hand from the Actions tab.

## Main functions

| Module | Function | What it does |
|---|---|---|
| `main.py` | `digest()` | Runs the daily pipeline: poll → ingest → rank → summarize → send. Papers that fail are returned to the pool. |
| | `alert(text)` | Sends an `[ALERT]` message to Telegram; never raises. |
| `sources.py` | `fetch_arxiv()`, `fetch_hf()` | Fetch new papers as `Paper` objects (arXiv IDs without version). |
| `ingest.py` | `ingest()` | Saves fetched papers (upsert by arXiv ID), then `prune()` deletes unpicked papers older than 7 days. |
| `rank.py` | `rank(conn)` | Embeds candidates (all-MiniLM-L6-v2), scores them, picks the top 5 spread across categories, and stores the features used. |
| | `unpick(conn, ids)` | Returns unsent papers to the candidate pool. |
| `ranker.py` | `build_features(...)` | The 5 ranking features: similarity to interests, liked papers and disliked papers, log upvotes, age. |
| | `active_model(conn)` | The active learned weights, or the hand-set defaults. |
| `summarize.py` | `Summarizer.summarize(title, text)` | One OpenRouter call (traced in Langfuse) returning a structured intro / results / discussion summary. |
| | `missing_numbers(summary, source)` | Flags numbers in the results summary that aren't in the source text. |
| | `describe_error(e)` | Readable one-line error for alerts, without secrets. |
| `telegram.py` | `Telegram.send_text(...)`, `get_updates(...)` | Bot API calls with 3 retries on connection errors. |
| | `format_card(...)`, `feedback_buttons(id)` | HTML-escaped paper card and its three buttons. |
| `poller.py` | `poll()` | Saves button taps to the `feedback` table (each tap once). |
| `retrain.py` | `retrain()` | With 100+ labels, trains logistic regression on the oldest 80%, tests on the newest 20%, and promotes it only if it beats the current ranker. Reports to Telegram either way. |
| `db.py` | `connect()` | Connects using `DATABASE_URL` and creates the tables if they're missing. |

## Setup

Secrets (GitHub → Settings → Secrets and variables → Actions, and `.env` locally, without quotes):
`DATABASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

```
pip install -r requirements.txt
python -m pytest            # tests, no network or keys needed
python -m src.main          # run a digest locally
pre-commit install          # gitleaks check on every commit
```

Settings to edit: `INTERESTS`, `TOP_N` (`src/rank.py`), `ARXIV_CATEGORIES` (`src/sources.py`), `MODEL` (`src/summarize.py`).
