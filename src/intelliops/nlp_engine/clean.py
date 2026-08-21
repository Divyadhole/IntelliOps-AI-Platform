"""Text normalisation for the feedback corpus."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_ORDER_ID = re.compile(r"\b(?:order|ticket|ref|case)[\s#:-]*[A-Z0-9]{4,}\b", re.I)
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_WHITESPACE = re.compile(r"\s+")

# Domain stopwords: high-frequency, zero-signal words in support text.
DOMAIN_STOPWORDS = {
    "customer", "service", "company", "please", "thanks", "thank", "hi", "hello",
    "regards", "team", "im", "ive", "dont", "didnt", "would", "could", "really",
    "also", "still", "just", "get", "got", "one", "even", "back", "since",
}


def normalise(text: str) -> str:
    """Lowercase, strip PII-ish tokens and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _EMAIL.sub(" <email> ", text)
    text = _URL.sub(" <url> ", text)
    text = _ORDER_ID.sub(" <ref> ", text)
    text = text.lower()
    text = _NUMBER.sub(" <num> ", text)
    text = re.sub(r"[^a-z<>\s']", " ", text)
    return _WHITESPACE.sub(" ", text).strip()


def clean_corpus(reviews: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Add a ``clean_text`` column and drop rows with nothing left to model."""
    out = reviews.copy()
    out["clean_text"] = out[text_column].map(normalise)
    out["token_count"] = out["clean_text"].str.split().str.len().fillna(0).astype(int)
    return out[out["token_count"] >= 3].reset_index(drop=True)
