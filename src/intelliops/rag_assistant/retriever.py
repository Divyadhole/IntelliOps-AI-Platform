"""Corpus construction: turn the warehouse into retrievable evidence.

The point of difference versus a toy RAG demo: the index contains *both* verbatim
customer text *and* rendered facts from the structured layer — KPIs, segment churn
rates, model drivers, topic aggregates. That is what lets one question
("why are customers leaving?") be answered with a number, a cause and a quote.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..config import Config, load_config
from ..data_pipeline import warehouse
from ..logging_utils import get_logger, stage
from .vector_store import Document, VectorStore

logger = get_logger(__name__)


def _kpi_documents(cfg: Config) -> list[Document]:
    docs: list[Document] = []
    try:
        kpis = warehouse.executive_kpis(cfg).iloc[0]
        docs.append(
            Document(
                doc_id="kpi::overall",
                source="kpi",
                text=(
                    f"Company-wide metrics: {int(kpis['customers']):,} active customers; "
                    f"churn rate {kpis['churn_rate_pct']}%; "
                    f"monthly recurring revenue ${kpis['monthly_recurring_revenue']:,.0f}; "
                    f"annualised revenue ${kpis['annualised_revenue']:,.0f}; "
                    f"average tenure {kpis['avg_tenure_months']} months; ARPU ${kpis['arpu']}."
                ),
                metadata=kpis.to_dict(),
            )
        )
    except Exception as exc:
        logger.warning("Could not build KPI documents: %s", exc)

    try:
        for _, row in warehouse.segment_risk(cfg).iterrows():
            docs.append(
                Document(
                    doc_id=f"kpi::contract::{row['contract_type']}",
                    source="kpi",
                    text=(
                        f"Contract segment '{row['contract_type']}': {int(row['customers']):,} customers, "
                        f"churn rate {row['churn_rate_pct']}%, "
                        f"${row['margin_at_risk']:,.0f} of annual gross margin at risk."
                    ),
                    metadata=row.to_dict(),
                )
            )
    except Exception as exc:
        logger.warning("Could not build segment documents: %s", exc)
    return docs


def _topic_documents(cfg: Config) -> list[Document]:
    if not warehouse.table_exists("agg_topic_summary", cfg):
        logger.warning("agg_topic_summary missing — run the NLP pipeline first for richer answers")
        return []
    summary = warehouse.read_table("agg_topic_summary", cfg)
    docs = []
    for _, row in summary.iterrows():
        docs.append(
            Document(
                doc_id=f"topic::{int(row['topic_id'])}",
                source="topic_summary",
                text=(
                    f"Feedback theme '{row['topic_label']}': {int(row['documents']):,} messages from "
                    f"{int(row['customers_affected']):,} customers, "
                    f"{row['negative_share'] * 100:.0f}% negative, "
                    f"average rating {row['avg_rating']:.2f}/5, "
                    f"average sentiment {row['avg_sentiment']:.2f}."
                ),
                metadata=row.to_dict(),
            )
        )
    return docs


def _model_driver_documents(cfg: Config) -> list[Document]:
    """Render the churn model's global SHAP drivers as retrievable statements."""
    try:
        import numpy as np

        from ..churn_model.explain import global_importance
        from ..churn_model.predict import ChurnScorer

        scorer = ChurnScorer(cfg=cfg)
        features = warehouse.read_table(cfg["warehouse.schema_tables.features"], cfg, limit=500)
        importance = global_importance(scorer.bundle, np.asarray(scorer.encode(features)), top_n=10)
        docs = []
        for rank, row in enumerate(importance.itertuples(), start=1):
            docs.append(
                Document(
                    doc_id=f"driver::{row.feature}",
                    source="model_driver",
                    text=(
                        f"Churn model driver #{rank}: '{row.feature}' with mean absolute SHAP "
                        f"contribution {row.importance:.4f}. Model: {scorer.bundle.get('model_name')}, "
                        f"test ROC-AUC {scorer.bundle.get('metrics', {}).get('roc_auc', float('nan')):.3f}."
                    ),
                    metadata={"feature": row.feature, "importance": float(row.importance), "rank": rank},
                )
            )
        return docs
    except Exception as exc:
        logger.warning("Could not build model-driver documents (%s) — train the model first", exc)
        return []


def _review_documents(cfg: Config, max_docs: int = 3000) -> list[Document]:
    table = "fct_review_nlp" if warehouse.table_exists("fct_review_nlp", cfg) \
        else cfg["warehouse.schema_tables.reviews"]
    reviews = warehouse.read_table(table, cfg)
    if len(reviews) > max_docs:
        # Bias the sample toward negative feedback: that is what analysts ask about.
        if "sentiment_score" in reviews.columns:
            reviews = reviews.sort_values("sentiment_score").head(max_docs)
        else:
            reviews = reviews.sample(max_docs, random_state=cfg.seed)
    docs = []
    for _, row in reviews.iterrows():
        meta = {k: row[k] for k in ("customerID", "rating", "channel", "sentiment_label", "topic_label")
                if k in reviews.columns}
        prefix = f"[{meta.get('channel', 'feedback')}, rating {meta.get('rating', '?')}/5] "
        docs.append(
            Document(doc_id=f"review::{row['review_id']}", source="review",
                     text=prefix + str(row["review_text"]), metadata=meta)
        )
    return docs


def build_corpus(cfg: Config | None = None) -> list[Document]:
    cfg = cfg or load_config()
    docs = _kpi_documents(cfg) + _topic_documents(cfg) + _model_driver_documents(cfg) + _review_documents(cfg)
    counts = pd.Series([d.source for d in docs]).value_counts().to_dict()
    logger.info("Corpus assembled: %d documents %s", len(docs), counts)
    return docs


def build_index(cfg: Config | None = None) -> VectorStore:
    cfg = cfg or load_config()
    with stage(logger, "build RAG index"):
        store = VectorStore(backend=cfg.get("rag.embedding_backend", "auto"), cfg=cfg)
        store.build(build_corpus(cfg))
        store.save(cfg.path("rag.vector_store"))
    return store


def load_index(cfg: Config | None = None) -> VectorStore:
    cfg = cfg or load_config()
    path = cfg.path("rag.vector_store")
    if not path.exists():
        logger.info("No index at %s — building one now", path)
        return build_index(cfg)
    return VectorStore.load(path, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG vector index")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build_index(load_config(args.config) if args.config else None)


if __name__ == "__main__":
    main()
