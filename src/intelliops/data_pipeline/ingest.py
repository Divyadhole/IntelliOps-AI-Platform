"""Ingestion layer: source-agnostic loading of structured and text data.

Resolution order for each source:
  1. the real CSV, if present at the configured path (Kaggle download);
  2. the synthetic generator, which emits an identical schema.

Every load is logged with a row/column count so lineage is traceable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger
from .synthetic import generate_customers, generate_reviews

logger = get_logger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded real source %s (%d rows, %d cols)", path.name, len(df), df.shape[1])
    return df


def load_customers(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.path("ingest.telco_csv")
    if path.exists():
        return _read_csv(path)
    n = int(cfg["ingest.synthetic_customers"])
    logger.warning("Real Telco CSV not found at %s — generating %d synthetic records", path, n)
    df = generate_customers(n=n, seed=cfg.seed)
    logger.info("Generated synthetic customers (%d rows, %d cols)", len(df), df.shape[1])
    return df


def load_reviews(customers: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Load the feedback corpus, keyed to the customers already ingested."""
    cfg = cfg or load_config()
    path = cfg.path("ingest.reviews_csv")
    if path.exists():
        return _read_csv(path)
    n = int(cfg["ingest.synthetic_reviews"])
    logger.warning("Real review CSV not found at %s — generating %d synthetic reviews", path, n)
    target = cfg.get("churn_model.target", "Churn")
    churn_flags = customers[target].tolist() if target in customers.columns else None
    return generate_reviews(customers["customerID"].tolist(), n=n, seed=cfg.seed, churn_flags=churn_flags)
