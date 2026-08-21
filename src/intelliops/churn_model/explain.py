"""Explainable AI: SHAP attributions translated into language a retention team can act on.

A raw SHAP bar chart is not an explanation to a business user. This module does two
things: computes the attributions, then *renders* the top drivers as sentences with
a recommended action attached, which is what makes the output usable in a CRM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Encoded feature name → (plain-English phrase, suggested action when it drives risk up)
FEATURE_NARRATIVES: dict[str, tuple[str, str]] = {
    "tenure": ("short tenure with the company", "Enrol in the first-year onboarding and check-in programme"),
    "MonthlyCharges": ("high monthly charges", "Review plan fit; offer a right-sized bundle"),
    "TotalCharges": ("low lifetime billing to date", "Too early to be loyal — prioritise early-life engagement"),
    "charge_drift": ("recent price increase versus their historical average",
                     "Grandfather the previous rate or offer a loyalty credit"),
    "charge_drift_pct": ("bill rising faster than their historical average", "Proactive bill-shock outreach"),
    "avg_charge_per_month": ("high average spend", "Confirm they are on the best available tariff"),
    "services_count": ("few products held", "Cross-sell a sticky add-on (security or support)"),
    "has_premium_support": ("no premium support attached", "Offer 3 months of TechSupport free"),
    "is_month_to_month": ("month-to-month contract", "Offer a discounted 12-month contract"),
    "is_manual_payment": ("manual payment method", "Incentivise switching to autopay"),
    "is_new_customer": ("still in the first six months", "Assign to the new-customer nurture track"),
    "avg_rating": ("low average feedback rating", "Route to a service-recovery specialist"),
    "min_rating": ("at least one very negative review", "Service recovery call within 48 hours"),
    "has_negative_review": ("logged a negative review", "Service recovery call within 48 hours"),
    "review_count": ("frequent support contact", "Investigate the underlying unresolved issue"),
    "Contract_Month-to-month": ("month-to-month contract", "Offer a discounted 12-month contract"),
    "Contract_Two year": ("two-year contract", "Low-risk — no action needed"),
    "InternetService_Fiber optic": ("fibre service (historically higher churn)",
                                    "Check line quality and recent outage history"),
    "PaymentMethod_Electronic check": ("paying by electronic check",
                                       "Offer a discount for switching to autopay"),
    "TechSupport_No": ("no tech-support package", "Offer 3 months of TechSupport free"),
    "OnlineSecurity_No": ("no online-security package", "Bundle security at no extra cost for 3 months"),
}


def _narrate(feature: str, direction: str) -> tuple[str, str]:
    """Map an encoded feature to (reason phrase, action). Falls back gracefully."""
    if feature in FEATURE_NARRATIVES:
        phrase, action = FEATURE_NARRATIVES[feature]
        return phrase, action
    if "_" in feature:
        base, _, level = feature.partition("_")
        if base in FEATURE_NARRATIVES:
            phrase, action = FEATURE_NARRATIVES[base]
            return f"{phrase} ({level})", action
        return f"{base.replace('_', ' ')} = {level}", "Review with the segment owner"
    pretty = feature.replace("_", " ")
    qualifier = "elevated" if direction == "increases risk" else "low"
    return f"{qualifier} {pretty}", "Review with the segment owner"


def build_explainer(bundle: dict[str, Any], background: np.ndarray | None = None):
    """Return a SHAP explainer for the bundle's base model, or None if SHAP is absent."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — explanations will fall back to model coefficients")
        return None

    model = bundle["base_model"]
    name = bundle.get("model_name", "")
    # Always prefer the stored training-distribution background over the request itself.
    reference = bundle.get("background")
    if reference is None:
        reference = background
    try:
        if name in {"xgboost", "lightgbm", "random_forest"}:
            return shap.TreeExplainer(model)
        if reference is not None:
            return shap.LinearExplainer(model, np.asarray(reference))
        return shap.Explainer(model)
    except Exception as exc:  # pragma: no cover - explainer construction is model-specific
        logger.warning("Could not build a SHAP explainer (%s) — falling back", exc)
        return None


def _shap_matrix(explainer, encoded: np.ndarray) -> np.ndarray | None:
    if explainer is None:
        return None
    try:
        values = explainer.shap_values(encoded)
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP computation failed (%s)", exc)
        return None
    values = np.asarray(values)
    if values.ndim == 3:            # (n, features, classes) or (classes, n, features)
        values = values[..., -1] if values.shape[-1] <= 3 else values[-1]
    return np.asarray(values, dtype=float)


def global_importance(bundle: dict[str, Any], encoded: np.ndarray, top_n: int = 20) -> pd.DataFrame:
    """Mean |SHAP| per feature — the global driver ranking."""
    explainer = build_explainer(bundle, background=encoded[:100])
    values = _shap_matrix(explainer, encoded)
    names = bundle["feature_names"]

    if values is None:  # fallback: coefficients or impurity importances
        model = bundle["base_model"]
        raw = getattr(model, "feature_importances_", None)
        if raw is None:
            coef = getattr(model, "coef_", None)
            raw = np.abs(coef[0]) if coef is not None else np.zeros(len(names))
        importance = np.asarray(raw, dtype=float)
        method = "model_importance"
    else:
        importance = np.abs(values).mean(axis=0)
        method = "mean_abs_shap"

    df = pd.DataFrame({"feature": names[: len(importance)], "importance": importance, "method": method})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def explain_customer(bundle: dict[str, Any], encoded_row: np.ndarray, top_n: int = 4,
                     cfg: Config | None = None) -> list[dict[str, Any]]:
    """Top drivers for a single prediction, rendered as reason + recommended action."""
    cfg = cfg or load_config()
    encoded_row = np.asarray(encoded_row).reshape(1, -1)
    explainer = build_explainer(bundle, background=encoded_row)
    values = _shap_matrix(explainer, encoded_row)
    names = bundle["feature_names"]

    if values is None:
        model = bundle["base_model"]
        raw = getattr(model, "feature_importances_", None)
        if raw is None:
            coef = getattr(model, "coef_", None)
            raw = coef[0] if coef is not None else np.zeros(len(names))
        contributions = np.asarray(raw, dtype=float) * encoded_row.ravel()[: len(raw)]
    else:
        contributions = values.ravel()

    order = np.argsort(-np.abs(contributions))[:top_n]
    drivers = []
    for i in order:
        if i >= len(names):
            continue
        contribution = float(contributions[i])
        direction = "increases risk" if contribution > 0 else "reduces risk"
        reason, action = _narrate(names[i], direction)
        drivers.append(
            {
                "feature": names[i],
                "contribution": round(contribution, 5),
                "direction": direction,
                "reason": reason,
                "recommended_action": action if contribution > 0 else "—",
            }
        )
    return drivers
