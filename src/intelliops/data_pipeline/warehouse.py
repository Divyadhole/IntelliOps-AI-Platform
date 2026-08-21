"""Warehouse layer — SQLAlchemy writes/reads against SQLite locally, Postgres in prod.

The only thing that changes between environments is ``INTELLIOPS_DB_URL``.
Analytical SQL lives here rather than in notebooks so the dashboard and the API
read the *same* definitions of "high risk" and "revenue at risk".
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_engine(db_url: str | None = None) -> Engine:
    cfg = load_config()
    url = db_url or cfg.db_url
    logger.info("Warehouse engine: %s", url.split("@")[-1])
    return create_engine(url, future=True)


def write_table(df: pd.DataFrame, table: str, cfg: Config | None = None,
                if_exists: str = "replace") -> int:
    cfg = cfg or load_config()
    engine = get_engine(cfg.db_url)
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    logger.info("Wrote %d rows → %s", len(df), table)
    return len(df)


def read_table(table: str, cfg: Config | None = None, limit: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    engine = get_engine(cfg.db_url)
    sql = f"SELECT * FROM {table}" + (f" LIMIT {int(limit)}" if limit else "")
    return pd.read_sql(text(sql), engine)


def query(sql: str, cfg: Config | None = None, **params) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_sql(text(sql), get_engine(cfg.db_url), params=params or None)


def table_exists(table: str, cfg: Config | None = None) -> bool:
    cfg = cfg or load_config()
    try:
        query(f"SELECT 1 FROM {table} LIMIT 1", cfg)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Analytical SQL: the shared definitions used by the dashboard, API and RAG layer
# --------------------------------------------------------------------------

EXEC_KPI_SQL = """
SELECT
    COUNT(*)                                            AS customers,
    ROUND(AVG(CAST(Churn AS FLOAT)) * 100, 2)           AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2)                       AS monthly_recurring_revenue,
    ROUND(SUM(MonthlyCharges) * 12, 2)                  AS annualised_revenue,
    ROUND(AVG(tenure), 1)                               AS avg_tenure_months,
    ROUND(AVG(MonthlyCharges), 2)                       AS arpu
FROM {features}
"""

SEGMENT_RISK_SQL = """
SELECT
    Contract                                            AS contract_type,
    COUNT(*)                                            AS customers,
    ROUND(AVG(CAST(Churn AS FLOAT)) * 100, 2)           AS churn_rate_pct,
    ROUND(SUM(annual_margin_at_risk), 2)                AS margin_at_risk
FROM {features}
GROUP BY Contract
ORDER BY churn_rate_pct DESC
"""

HIGH_RISK_SQL = """
SELECT
    p.customerID,
    ROUND(p.churn_probability, 4)                       AS churn_probability,
    p.risk_band,
    p.recommended_action,
    ROUND(p.expected_value_of_offer, 2)                 AS expected_value_of_offer,
    f.tenure,
    f.MonthlyCharges,
    f.Contract,
    ROUND(f.annual_margin_at_risk, 2)                   AS annual_margin_at_risk
FROM {predictions} p
JOIN {features} f ON f.customerID = p.customerID
WHERE p.churn_probability >= :threshold
ORDER BY p.expected_value_of_offer DESC
LIMIT :limit
"""


def executive_kpis(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return query(EXEC_KPI_SQL.format(features=cfg["warehouse.schema_tables.features"]), cfg)


def segment_risk(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return query(SEGMENT_RISK_SQL.format(features=cfg["warehouse.schema_tables.features"]), cfg)


def high_risk_customers(threshold: float = 0.55, limit: int = 100,
                        cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    sql = HIGH_RISK_SQL.format(
        predictions=cfg["warehouse.schema_tables.predictions"],
        features=cfg["warehouse.schema_tables.features"],
    )
    return query(sql, cfg, threshold=float(threshold), limit=int(limit))
