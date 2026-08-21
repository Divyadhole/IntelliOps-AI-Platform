"""Sentiment scoring with a transformer when available, lexicon otherwise.

The lexicon path exists so the platform never hard-fails on a machine without
``transformers`` or a GPU — the module degrades in quality, not in availability,
and it reports which backend produced each score so nobody misreads the results.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)

_NEGATIVE = {
    "unusable": 2.0, "unacceptable": 2.0, "terrible": 2.0, "awful": 2.0, "worst": 2.0,
    "slow": 1.2, "late": 1.2, "delay": 1.3, "delayed": 1.3, "dropping": 1.2, "outage": 1.6,
    "outages": 1.6, "charged": 1.0, "overcharged": 1.8, "mess": 1.4, "broken": 1.6,
    "poor": 1.4, "bad": 1.3, "complain": 1.3, "complained": 1.3, "cancel": 1.7,
    "cancelled": 1.5, "refund": 1.2, "waiting": 1.0, "never": 1.0, "again": 0.4,
    "painfully": 1.5, "problem": 1.2, "issue": 0.9, "fail": 1.6, "failed": 1.6,
}
_POSITIVE = {
    "excellent": 2.0, "great": 1.6, "good": 1.2, "reliable": 1.5, "solid": 1.4,
    "fast": 1.2, "quick": 1.2, "helpful": 1.5, "love": 1.7, "perfect": 1.8,
    "resolved": 1.4, "fixed": 1.3, "happy": 1.5, "recommend": 1.4, "value": 1.0,
    "better": 1.0, "smooth": 1.3, "easy": 1.1,
}
_NEGATORS = {"not", "no", "never", "cannot", "cant", "dont", "didnt", "wasnt", "isnt"}


def lexicon_sentiment(text: str) -> float:
    """Return a score in [-1, 1] using a weighted lexicon with negation handling."""
    tokens = text.split()
    score = 0.0
    weight = 0.0
    for i, token in enumerate(tokens):
        polarity = _POSITIVE.get(token, 0.0) - _NEGATIVE.get(token, 0.0)
        if polarity == 0.0:
            continue
        # a negator within the preceding two tokens flips the polarity
        if any(t in _NEGATORS for t in tokens[max(0, i - 2): i]):
            polarity = -polarity * 0.8
        score += polarity
        weight += abs(polarity)
    if weight == 0.0:
        return 0.0
    return float(np.clip(score / weight, -1.0, 1.0))


@lru_cache(maxsize=1)
def _transformer_pipeline(model_name: str):
    try:
        from transformers import pipeline  # type: ignore

        logger.info("Loading transformer sentiment model: %s", model_name)
        return pipeline("sentiment-analysis", model=model_name, truncation=True)
    except Exception as exc:
        logger.warning("Transformer sentiment unavailable (%s) — using lexicon backend", exc.__class__.__name__)
        return None


def score_sentiment(texts: pd.Series, cfg: Config | None = None) -> pd.DataFrame:
    """Score a series of cleaned texts. Returns score, label and the backend used."""
    cfg = cfg or load_config()
    backend = cfg.get("nlp.sentiment_backend", "auto")
    pipe = None
    if backend in {"auto", "transformer"}:
        pipe = _transformer_pipeline(cfg.get("nlp.transformer_model"))
        if pipe is None and backend == "transformer":
            logger.warning("sentiment_backend=transformer requested but unavailable; falling back")

    if pipe is not None:
        raw = pipe(texts.tolist(), batch_size=32)
        scores = [
            r["score"] if r["label"].upper().startswith("POS") else -r["score"] for r in raw
        ]
        used = "transformer"
    else:
        scores = [lexicon_sentiment(t) for t in texts]
        used = "lexicon"

    scores = np.asarray(scores, dtype=float)
    labels = np.where(scores > 0.15, "positive", np.where(scores < -0.15, "negative", "neutral"))
    logger.info("Sentiment (%s backend): %s", used,
                dict(pd.Series(labels).value_counts()))
    return pd.DataFrame({"sentiment_score": scores, "sentiment_label": labels, "sentiment_backend": used})
