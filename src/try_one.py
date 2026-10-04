import logging
from dotenv import load_dotenv
from sources import fetch_arxiv
from summarize import Summarizer, MODEL
from telegram import Telegram, format_card, feedback_buttons

logging.basicConfig(level=logging.INFO)
load_dotenv()

paper = fetch_arxiv(max_results=1)[0]
s = Summarizer().summarize(paper.title, paper.abstract)
text = "\n\n".join([s.intro.short, s.results.short, s.discussion.short])
Telegram().send_text(format_card(paper.arxiv_id, paper.title, text, MODEL),
                     reply_markup=feedback_buttons(paper.arxiv_id))
