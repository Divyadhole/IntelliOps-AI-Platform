"""Pipeline orchestrator: raw → validated → cleaned → featurised → warehouse.

Run directly (``python -m intelliops.data_pipeline.run_pipeline``) or as an Airflow
task — ``run()`` is a pure function with no CLI coupling, which is what makes it
schedulable. See ``airflow_dag.py`` for the DAG wrapper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger, stage
from . import warehouse
from .ingest import load_customers, load_reviews
from .transform import aggregate_review_features, clean_customers, engineer_features
from .validate import validate_customers

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    customers: pd.DataFrame
    features: pd.DataFrame
    reviews: pd.DataFrame
    quality: pd.DataFrame

    def summary(self) -> dict[str, int]:
        return {
            "customers": len(self.customers),
            "features_rows": len(self.features),
            "features_cols": self.features.shape[1],
            "reviews": len(self.reviews),
            "quality_checks": len(self.quality),
        }


def _snapshot(df: pd.DataFrame, cfg: Config, name: str) -> None:
    """Write a columnar snapshot for downstream jobs; fall back to CSV without pyarrow."""
    out_dir = cfg.path("paths.data_processed")
    try:
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    except (ImportError, ValueError) as exc:
        logger.warning("Parquet unavailable (%s) — writing %s.csv instead", exc.__class__.__name__, name)
        df.to_csv(out_dir / f"{name}.csv", index=False)


def run(cfg: Config | None = None, persist: bool = True) -> PipelineResult:
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    tables = cfg["warehouse.schema_tables"]

    with stage(logger, "1/5 ingest"):
        raw = load_customers(cfg)

    with stage(logger, "2/5 validate (data contract)"):
        report = validate_customers(raw, cfg)
        quality = report.to_frame()
        quality["run_ts"] = pd.Timestamp.now(tz="UTC").isoformat()

    with stage(logger, "3/5 clean"):
        clean = clean_customers(raw, cfg)

    with stage(logger, "4/5 feature engineering"):
        features = engineer_features(clean, cfg)
        reviews = load_reviews(clean, cfg)
        review_feats = aggregate_review_features(reviews)
        if not review_feats.empty:
            features = features.merge(review_feats, on="customerID", how="left")
            features["review_count"] = features["review_count"].fillna(0).astype(int)
            features["avg_rating"] = features["avg_rating"].fillna(features["avg_rating"].median())
            features["min_rating"] = features["min_rating"].fillna(features["avg_rating"])
            features["has_negative_review"] = features["has_negative_review"].fillna(0).astype(int)
            features = features.drop(columns=["last_review_date"], errors="ignore")

    with stage(logger, "5/5 load to warehouse"):
        if persist:
            warehouse.write_table(clean, tables["customers"], cfg)
            warehouse.write_table(features, tables["features"], cfg)
            warehouse.write_table(reviews, tables["reviews"], cfg)
            warehouse.write_table(quality, tables["quality"], cfg)
            _snapshot(features, cfg, "features")
            _snapshot(reviews, cfg, "reviews")
        else:
            logger.info("persist=False — skipping warehouse writes")

    result = PipelineResult(clean, features, reviews, quality)
    logger.info("Pipeline summary: %s", result.summary())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IntelliOps ETL pipeline")
    parser.add_argument("--no-persist", action="store_true", help="run without writing to the warehouse")
    parser.add_argument("--config", default=None, help="path to an alternative config.yaml")
    args = parser.parse_args()
    run(load_config(args.config) if args.config else None, persist=not args.no_persist)


if __name__ == "__main__":
    main()
