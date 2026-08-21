"""A small, dependency-light vector store with a pluggable embedding backend.

Backends, in preference order:
  1. ``sentence-transformers`` (all-MiniLM-L6-v2) — real dense embeddings;
  2. TF-IDF + truncated SVD — a deterministic LSA fallback that needs nothing extra.

Retrieval is **hybrid**: dense cosine similarity fused with lexical TF-IDF scores.
Pure dense retrieval loses exact matches on identifiers and product names, which is
exactly what a business analyst asks about ("what's happening with fibre?").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    doc_id: str
    text: str
    source: str                       # e.g. "review", "topic_summary", "kpi", "model_driver"
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Embeds documents once, then answers hybrid similarity queries."""

    def __init__(self, backend: str = "auto", model_name: str | None = None, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.requested_backend = backend
        self.model_name = model_name or self.cfg.get("rag.embedding_model")
        self.backend: str = "tfidf"
        self.documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        self._encoder = None
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._svd: TruncatedSVD | None = None

    # ------------------------------------------------------------- encoding
    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading embedding model: %s", self.model_name)
            return SentenceTransformer(self.model_name)
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s) — using TF-IDF+SVD embeddings",
                           exc.__class__.__name__)
            return None

    def build(self, documents: list[Document]) -> VectorStore:
        if not documents:
            raise ValueError("Cannot build a vector store from zero documents")
        self.documents = documents
        texts = [d.text for d in documents]

        if self.requested_backend in {"auto", "sentence_transformers"}:
            self._encoder = self._load_sentence_transformer()

        # Lexical index is always built — it is half of the hybrid score.
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=20000)
        self._tfidf_matrix = self._tfidf.fit_transform(texts)

        if self._encoder is not None:
            self.backend = "sentence_transformers"
            dense = np.asarray(self._encoder.encode(texts, batch_size=64, show_progress_bar=False))
        else:
            self.backend = "tfidf_svd"
            n_components = int(min(256, max(2, self._tfidf_matrix.shape[1] - 1, 2)))
            n_components = min(n_components, self._tfidf_matrix.shape[0] - 1) if len(texts) > 2 else 2
            self._svd = TruncatedSVD(n_components=n_components, random_state=self.cfg.seed)
            dense = self._svd.fit_transform(self._tfidf_matrix)

        self._embeddings = normalize(np.asarray(dense, dtype=np.float32))
        logger.info("Vector store built: %d documents, backend=%s, dim=%d",
                    len(documents), self.backend, self._embeddings.shape[1])
        return self

    def _embed_query(self, query: str) -> np.ndarray:
        if self._encoder is not None:
            vec = np.asarray(self._encoder.encode([query]))
        else:
            vec = self._svd.transform(self._tfidf.transform([query]))
        return normalize(np.asarray(vec, dtype=np.float32))

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, top_k: int = 6, alpha: float = 0.65,
               sources: list[str] | None = None) -> list[tuple[Document, float]]:
        """Hybrid search. ``alpha`` weights dense similarity against lexical overlap."""
        if self._embeddings is None:
            raise RuntimeError("VectorStore.build() must be called before search()")

        dense_scores = (self._embeddings @ self._embed_query(query).T).ravel()
        lexical = (self._tfidf_matrix @ self._tfidf.transform([query]).T).toarray().ravel()
        if lexical.max() > 0:
            lexical = lexical / lexical.max()
        scores = alpha * dense_scores + (1 - alpha) * lexical

        if sources:
            mask = np.array([d.source in sources for d in self.documents])
            scores = np.where(mask, scores, -np.inf)

        order = np.argsort(-scores)[:top_k]
        return [(self.documents[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def search_stratified(self, query: str, quotas: dict[str, int],
                          alpha: float = 0.65) -> list[tuple[Document, float]]:
        """Retrieve a fixed budget per source.

        A flat top-k over this corpus is dominated by whichever source has the most
        near-duplicate documents (3,000 reviews, 8 topic rows). Stratifying the budget
        guarantees an executive answer sees the KPI, the model's drivers, the feedback
        themes *and* verbatim quotes rather than six variations of one of them.
        """
        results: list[tuple[Document, float]] = []
        for source, k in quotas.items():
            if k <= 0:
                continue
            results.extend(self.search(query, top_k=k, alpha=alpha, sources=[source]))
        return sorted(results, key=lambda pair: -pair[1])

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The sentence-transformer itself is not pickled — it is reloaded by name.
        joblib.dump(
            {
                "documents": self.documents,
                "embeddings": self._embeddings,
                "tfidf": self._tfidf,
                "tfidf_matrix": self._tfidf_matrix,
                "svd": self._svd,
                "backend": self.backend,
                "model_name": self.model_name,
            },
            path,
        )
        logger.info("Saved vector store → %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path, cfg: Config | None = None) -> VectorStore:
        state = joblib.load(Path(path))
        store = cls(backend="auto", model_name=state.get("model_name"), cfg=cfg)
        store.documents = state["documents"]
        store._embeddings = state["embeddings"]
        store._tfidf = state["tfidf"]
        store._tfidf_matrix = state["tfidf_matrix"]
        store._svd = state["svd"]
        store.backend = state["backend"]
        if store.backend == "sentence_transformers":
            store._encoder = store._load_sentence_transformer()
            if store._encoder is None:  # model no longer available on this machine
                raise RuntimeError("Vector store was built with sentence-transformers, which is now missing. "
                                   "Rebuild with `make rag-index`.")
        logger.info("Loaded vector store: %d documents (backend=%s)", len(store.documents), store.backend)
        return store
