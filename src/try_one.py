"""Print a fresh summary of one paper without sending it or touching the database.

    python -m src.try_one 2610.11959   # a paper already in the DB, e.g. one you were sent
    python -m src.try_one              # the newest arXiv paper
"""
import sys
import logging
from dotenv import load_dotenv
from src.db import connect
from src.sources import fetch_arxiv, paper_text
from src.summarize import Summarizer, missing_numbers


def main(argv: list[str]) -> None:
    if argv:
        with connect() as conn:
            row = conn.execute("select arxiv_id, title, abstract from papers where arxiv_id = %s",
                               (argv[0],)).fetchone()
        if row is None:
            sys.exit(f"{argv[0]} is not in the papers table")
        arxiv_id, title, abstract = row
    else:
        p = fetch_arxiv(max_results=1)[0]
        arxiv_id, title, abstract = p.arxiv_id, p.title, p.abstract

    text, source = paper_text(arxiv_id, abstract)
    s = Summarizer().summarize(title, text)
    print(f"{title}\nhttps://arxiv.org/abs/{arxiv_id}\nsource: {source}\n")
    print(f"{s.intro}\n\n{s.results}\n\n{s.discussion}")
    if missing := missing_numbers(s, text):
        print(f"\nnumbers not found in the source: {missing}")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
