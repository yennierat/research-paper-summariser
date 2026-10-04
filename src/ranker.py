import math
import numpy as np

FEATURES = ["sim_profile", "sim_liked", "sim_disliked", "log_upvotes", "age_days"]
DEFAULT_WEIGHTS = np.array([1.0, 0.5, -0.5, 0.1, 0.0])
POSITIVE = {"like", "save"}

LATEST_FEEDBACK = """
    select distinct on (arxiv_id) arxiv_id, label, update_id
    from feedback order by arxiv_id, update_id desc
"""


def build_features(emb, profile, liked, disliked, hf_upvotes, age_days) -> list[float]:
    """Used at selection time only; training reads the stored result, so the two can't drift."""
    sim = lambda c: float(emb @ c) if c is not None else 0.0
    return [sim(profile), sim(liked), sim(disliked), math.log1p(hf_upvotes or 0), age_days]


def active_model(conn) -> tuple[np.ndarray, float]:
    row = conn.execute("select weights, bias from models where active").fetchone()
    return (np.array(row[0]), row[1]) if row else (DEFAULT_WEIGHTS, 0.0)
