import logging
from datetime import datetime, timezone
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ranker import build_features, active_model, LATEST_FEEDBACK, POSITIVE
from src.sources import ARXIV_CATEGORIES

log = logging.getLogger(__name__)

INTERESTS = (
    "Large language models and LLMOps / MLOps: training, fine-tuning, evaluation, serving and "
    "deploying models in production. AI for climate change, weather and environmental science. "
    "Computer vision. LLM security and cybersecurity: jailbreaks, prompt injection, red teaming, "
    "and using LLMs for security tasks."
)
TOP_N = 5
CANDIDATE_DAYS = 3


def _centroid(vectors):
    if not vectors:
        return None
    c = np.mean(vectors, axis=0)
    return c / np.linalg.norm(c)


def pick(order: list[int], categories: list[list[str]], n: int) -> list[int]:
    """Best unpicked paper per arXiv category, topped up from the overall order, best first."""
    picks = []
    for cat in ARXIV_CATEGORIES:
        best = next((i for i in order if cat in categories[i] and i not in picks), None)
        if best is not None:
            picks.append(best)
    picks += [i for i in order if i not in picks][:n - len(picks)]
    return sorted(picks[:n], key=order.index)


def rank(conn, n: int = TOP_N) -> list[str]:
    candidates = conn.execute(
        "select arxiv_id, title, abstract, hf_upvotes, published, categories from papers"
        " where selected_at is null and published > now() - make_interval(days => %s)",
        (CANDIDATE_DAYS,),
    ).fetchall()
    if not candidates:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode([f"{c[1]}. {c[2]}" for c in candidates], normalize_embeddings=True)
    profile = model.encode(INTERESTS, normalize_embeddings=True)

    rated = conn.execute(
        f"select p.embedding, f.label from ({LATEST_FEEDBACK}) f"
        " join papers p using (arxiv_id) where p.embedding is not null"
    ).fetchall()
    liked = _centroid([e for e, label in rated if label in POSITIVE])
    disliked = _centroid([e for e, label in rated if label not in POSITIVE])

    now = datetime.now(timezone.utc)
    X = np.array([
        build_features(e, profile, liked, disliked, c[3], (now - c[4]).total_seconds() / 86400)
        for e, c in zip(embs, candidates)
    ])
    weights, bias = active_model(conn)
    order = list(np.argsort(X @ weights + bias)[::-1])
    picks = pick(order, [c[5] for c in candidates], n)

    for i in picks:
        conn.execute(
            "update papers set embedding = %s, features = %s, selected_at = now() where arxiv_id = %s",
            (embs[i].tolist(), X[i].tolist(), candidates[i][0]),
        )
    conn.commit()
    log.info("ranked %d candidates, picked %d", len(candidates), len(picks))
    return [candidates[i][0] for i in picks]


def unpick(conn, arxiv_ids: list[str]) -> None:
    """Undo rank() for papers that never reached the user, so they can be picked again."""
    conn.execute(
        "update papers set embedding = null, features = null, selected_at = null where arxiv_id = any(%s)",
        (arxiv_ids,),
    )
    conn.commit()
    log.info("returned %d unsent papers to the candidate pool", len(arxiv_ids))
