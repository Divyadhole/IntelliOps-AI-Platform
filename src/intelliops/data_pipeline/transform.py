"""Cleaning and feature engineering — the layer that turns raw rows into a feature table.

The engineered features are deliberately *business-legible*: every one of them can
be explained to a retention manager in a sentence, which is what makes the SHAP
output in Module 2 actionable rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def coerce_and_repair(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Row-wise cleaning only: type coercion, known-bad-column repair, null policy.

    Deliberately excludes deduplication. This function runs at **both** training and
    inference time, and dropping rows during inference would silently return fewer
    predictions than the caller asked for — a batch of two customers who both omit
    an ID must come back as two scores, not one.
    """
    cfg = cfg or load_config()
    df = df.copy()

    # TotalCharges ships as text with blanks for brand-new customers.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip().replace({"": np.nan}), errors="coerce"
        )
        blank = int(df["TotalCharges"].isna().sum())
        # A tenure-0 customer has genuinely billed nothing yet — impute 0, not the median.
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        logger.info("Repaired %d blank TotalCharges values via tenure × monthly reconstruction", blank)

    for col in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical nulls become an explicit level; silent dropping loses signal.
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    # Target → 0/1 regardless of how the source encoded it ("Yes"/"No", "1"/"0", bool).
    target = cfg.get("churn_model.target", "Churn")
    if target in df.columns and not pd.api.types.is_integer_dtype(df[target]):
        as_text = df[target].astype("string").str.strip().str.lower()
        df[target] = as_text.isin(["yes", "y", "true", "1", "churned"]).astype(int)

    return df.reset_index(drop=True)


def clean_customers(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """ETL-time cleaning: deduplicate, then coerce and repair."""
    before = len(df)
    df = df.drop_duplicates().copy()
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"], keep="first")
    logger.info("Deduplicated: %d → %d rows", before, len(df))
    return coerce_and_repair(df, cfg)


def prepare_for_inference(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Serving-time equivalent of the training transform: repair + engineer, no row loss."""
    return engineer_features(coerce_and_repair(df, cfg), cfg)


def engineer_features(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add derived behavioural and financial features."""
    cfg = cfg or load_config()
    out = df.copy()

    # A serving payload may legitimately omit lifetime billing (a CRM that only
    # knows the current plan). Reconstruct it rather than failing the request.
    if "TotalCharges" not in out.columns:
        out["TotalCharges"] = out["MonthlyCharges"] * out["tenure"]
        logger.debug("TotalCharges absent — reconstructed from tenure × MonthlyCharges")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"] * out["tenure"]).fillna(0.0)

    # --- financial behaviour ---------------------------------------------
    out["avg_charge_per_month"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], out["MonthlyCharges"]
    )
    # Positive = customer is paying more now than their historical average → recent price hike.
    out["charge_drift"] = out["MonthlyCharges"] - out["avg_charge_per_month"]
    out["charge_drift_pct"] = out["charge_drift"] / out["avg_charge_per_month"].replace(0, np.nan)
    out["charge_drift_pct"] = out["charge_drift_pct"].fillna(0.0)
    out["clv_to_date"] = out["TotalCharges"]

    # --- product depth ----------------------------------------------------
    present = [c for c in SERVICE_COLUMNS if c in out.columns]
    out["services_count"] = sum((out[c] == "Yes").astype(int) for c in present)
    premium = pd.Series(False, index=out.index)
    for col in ("TechSupport", "OnlineSecurity"):
        if col in out.columns:
            premium |= out[col] == "Yes"
    out["has_premium_support"] = premium.astype(int)

    # --- lifecycle --------------------------------------------------------
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)
    out["is_new_customer"] = (out["tenure"] <= 6).astype(int)

    # --- contract / payment risk flags -----------------------------------
    if "Contract" in out.columns:
        out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    if "PaymentMethod" in out.columns:
        out["is_manual_payment"] = out["PaymentMethod"].isin(
            ["Electronic check", "Mailed check"]
        ).astype(int)

    # --- revenue at risk (used directly by the executive dashboard) -------
    horizon = float(cfg.get("churn_model.economics.horizon_months", 12))
    margin = float(cfg.get("churn_model.economics.monthly_margin_multiplier", 0.35))
    out["annual_revenue_at_risk"] = out["MonthlyCharges"] * horizon
    out["annual_margin_at_risk"] = out["annual_revenue_at_risk"] * margin

    new_cols = [c for c in out.columns if c not in df.columns]
    logger.info("Engineered %d features: %s", len(new_cols), ", ".join(new_cols))
    return out


def aggregate_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """Roll the text corpus up to one row per customer for joining onto features."""
    if reviews.empty:
        return pd.DataFrame(columns=["customerID"])
    agg = (
        reviews.groupby("customerID")
        .agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean"),
            min_rating=("rating", "min"),
            last_review_date=("review_date", "max"),
        )
        .reset_index()
    )
    agg["has_negative_review"] = (agg["min_rating"] <= 2).astype(int)
    logger.info("Aggregated review features for %d customers", len(agg))
    return agg
