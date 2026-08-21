"""NLP pipeline entry point: clean → sentiment → topics → warehouse.

Output tables:
  * ``fct_review_nlp``    one row per review, enriched with sentiment and topic
  * ``agg_topic_summary`` one row per topic, with volume, sentiment and reach
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from ..config import Config, load_config
from ..data_pipeline import warehouse
from ..logging_utils import get_logger, stage
from .clean import clean_corpus
from .sentiment import score_sentiment
from .topics import assign_topics, fit_topics, topic_summary

logger = get_logger(__name__)


def run(cfg: Config | None = None, reviews: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    cfg.ensure_dirs()

    with stage(logger, "load review corpus"):
        if reviews is None:
            reviews = warehouse.read_table(cfg["warehouse.schema_tables.reviews"], cfg)
        logger.info("Corpus: %d documents", len(reviews))

    with stage(logger, "clean text"):
        cleaned = clean_corpus(reviews, cfg.get("nlp.text_column", "review_text"))
        logger.info("Retained %d documents after cleaning (median %d tokens)",
                    len(cleaned), int(cleaned["token_count"].median()))

    with stage(logger, "sentiment analysis"):
        sentiment = score_sentiment(cleaned["clean_text"], cfg)
        enriched = pd.concat([cleaned.reset_index(drop=True), sentiment], axis=1)

    with stage(logger, "topic discovery (TF-IDF → NMF)"):
        topic_model = fit_topics(enriched["clean_text"], cfg)
        topics = assign_topics(topic_model, enriched["clean_text"])
        enriched = pd.concat([enriched, topics], axis=1)

    with stage(logger, "aggregate and persist"):
        summary = topic_summary(enriched)
        persist_cols = [
            "review_id", "customerID", "review_date", "channel", "rating", "review_text",
            "clean_text", "sentiment_score", "sentiment_label", "sentiment_backend",
            "topic_id", "topic_label", "topic_strength",
        ]
        out = enriched[[c for c in persist_cols if c in enriched.columns]].copy()
        if "review_date" in out.columns:
            out["review_date"] = out["review_date"].astype(str)
        warehouse.write_table(out, "fct_review_nlp", cfg)
        warehouse.write_table(summary, "agg_topic_summary", cfg)

        joblib.dump(topic_model, cfg.path("paths.models") / "topic_model.joblib")
        (cfg.path("paths.reports") / "topic_summary.json").write_text(
            json.dumps(
                {
                    "topics": summary.to_dict(orient="records"),
                    "terms": {str(k): v for k, v in topic_model.topic_terms.items()},
                    "sentiment_backend": str(sentiment["sentiment_backend"].iloc[0]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    logger.info("Top pain points:\n%s", summary.head(5).to_string(index=False))
    return {"reviews": out, "topics": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NLP intelligence pipeline")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config) if args.config else None)


if __name__ == "__main__":
    main()
