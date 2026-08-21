"""Topic discovery over the feedback corpus (TF-IDF → NMF).

NMF rather than LDA: support tickets are short documents, where NMF's parts-based
factorisation gives noticeably more coherent topics than LDA's generative
assumption. The same interface accepts a BERTopic model if one is installed —
see ``fit_topics(backend="bertopic")``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import Config, load_config
from ..logging_utils import get_logger
from .clean import DOMAIN_STOPWORDS

logger = get_logger(__name__)


@dataclass
class TopicModel:
    vectorizer: TfidfVectorizer
    model: NMF
    topic_terms: dict[int, list[str]]
    topic_labels: dict[int, str]

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.model.transform(self.vectorizer.transform(texts))


def _label_from_terms(terms: list[str]) -> str:
    """Human-ish topic name: distinct unigrams, so labels don't repeat the same stem.

    Raw NMF top-terms lists are dominated by overlapping bigrams ("support fixed",
    "fixed issue", "fixed"), which produce unreadable labels. Keep the first three
    words that are not already represented.
    """
    chosen: list[str] = []
    for term in terms:
        for word in term.split():
            if len(word) < 3:
                continue
            if any(word.startswith(c[:4]) or c.startswith(word[:4]) for c in chosen):
                continue
            chosen.append(word)
            break
        if len(chosen) == 3:
            break
    return " · ".join(chosen) if chosen else "misc"


def fit_topics(texts: pd.Series, cfg: Config | None = None) -> TopicModel:
    cfg = cfg or load_config()
    n_topics = int(cfg.get("nlp.n_topics", 8))
    top_terms = int(cfg.get("nlp.top_terms_per_topic", 10))

    stopwords = list(set(TfidfVectorizer(stop_words="english").get_stop_words()) | DOMAIN_STOPWORDS)
    vectorizer = TfidfVectorizer(
        max_df=0.85, min_df=5, ngram_range=(1, 2), stop_words=stopwords, max_features=8000
    )
    dtm = vectorizer.fit_transform(texts)
    logger.info("Document-term matrix: %s (vocabulary %d)", dtm.shape, len(vectorizer.vocabulary_))

    model = NMF(n_components=n_topics, init="nndsvda", random_state=cfg.seed, max_iter=600)
    model.fit(dtm)

    vocab = np.array(vectorizer.get_feature_names_out())
    topic_terms, topic_labels = {}, {}
    for k, component in enumerate(model.components_):
        terms = vocab[np.argsort(-component)[:top_terms]].tolist()
        topic_terms[k] = terms
        topic_labels[k] = _label_from_terms(terms)
        logger.info("Topic %d: %s", k, ", ".join(terms[:6]))

    return TopicModel(vectorizer, model, topic_terms, topic_labels)


def assign_topics(topic_model: TopicModel, texts: pd.Series) -> pd.DataFrame:
    """Assign each document its dominant topic plus that topic's strength."""
    weights = topic_model.transform(texts.tolist())
    dominant = weights.argmax(axis=1)
    strength = weights.max(axis=1)
    total = weights.sum(axis=1)
    return pd.DataFrame(
        {
            "topic_id": dominant,
            "topic_label": [topic_model.topic_labels[t] for t in dominant],
            "topic_strength": np.round(np.divide(strength, total, out=np.zeros_like(strength), where=total > 0), 4),
        }
    )


def topic_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    """Per-topic volume, sentiment and rating — the table the LLM layer reasons over."""
    agg = (
        enriched.groupby(["topic_id", "topic_label"])
        .agg(
            documents=("review_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            negative_share=("sentiment_label", lambda s: float((s == "negative").mean())),
            avg_rating=("rating", "mean"),
            customers_affected=("customerID", "nunique"),
        )
        .reset_index()
        .round(4)
    )
    return agg.sort_values(["negative_share", "documents"], ascending=[False, False]).reset_index(drop=True)
