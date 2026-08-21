"""Evaluation and the business decision layer.

Two distinct questions, deliberately kept separate:

1. *How good is the ranking?* — ROC-AUC, PR-AUC, log-loss, Brier.
2. *Who should we actually call?* — an expected-value policy, because a retention
   campaign has a per-contact cost and a realistic save rate. A model that is
   excellent at ranking can still lose money if the threshold is picked by habit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import Config, load_config


def classification_report_dict(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-free ranking metrics plus point metrics at a given cut."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def lift_table(y_true, y_proba, n_bins: int = 10) -> list[dict[str, float]]:
    """Decile lift — the table a retention manager actually reads."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    base_rate = y_true.mean()
    rows, bins = [], np.array_split(np.arange(len(y_sorted)), n_bins)
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        decile_rate = float(y_sorted[idx].mean())
        rows.append(
            {
                "decile": i,
                "customers": int(len(idx)),
                "churn_rate": round(decile_rate, 4),
                "lift": round(decile_rate / base_rate, 3) if base_rate else 0.0,
                "cumulative_capture": round(float(y_sorted[: idx[-1] + 1].sum() / y_true.sum()), 4),
            }
        )
    return rows


@dataclass
class ThresholdPolicy:
    """The retention policy implied by the model plus campaign economics."""

    threshold: float
    n_targeted: int
    expected_net_saving: float
    campaign_cost: float
    roi: float
    offer_cost: float
    offer_success_rate: float
    horizon_months: int
    precision_at_threshold: float
    recall_at_threshold: float


def expected_value_of_offer(churn_proba, monthly_charges, cfg: Config | None = None) -> np.ndarray:
    """Expected $ gain from making a retention offer to each customer.

        EV = P(churn) × P(save | offer) × margin_at_risk − offer_cost

    Customers with EV ≤ 0 should not be contacted, no matter how high their risk:
    a low-margin customer is not worth a $45 offer.
    """
    cfg = cfg or load_config()
    econ = cfg["churn_model.economics"]
    margin_at_risk = (
        np.asarray(monthly_charges, dtype=float)
        * float(econ["monthly_margin_multiplier"])
        * float(econ["horizon_months"])
    )
    return (
        np.asarray(churn_proba, dtype=float)
        * float(econ["offer_success_rate"])
        * margin_at_risk
        - float(econ["retention_offer_cost"])
    )


def choose_business_threshold(y_true, y_proba, monthly_charges, cfg: Config | None = None,
                              grid: np.ndarray | None = None) -> ThresholdPolicy:
    """Sweep probability thresholds and keep the one maximising expected net saving."""
    cfg = cfg or load_config()
    econ = cfg["churn_model.economics"]
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    grid = np.arange(0.05, 0.96, 0.01) if grid is None else grid

    ev = expected_value_of_offer(y_proba, monthly_charges, cfg)
    best = None
    for t in grid:
        targeted = y_proba >= t
        n = int(targeted.sum())
        if n == 0:
            continue
        net = float(ev[targeted].sum())
        cost = n * float(econ["retention_offer_cost"])
        if best is None or net > best[1]:
            best = (float(t), net, n, cost)

    if best is None:  # degenerate case: nobody scored above the lowest threshold
        best = (0.5, 0.0, 0, 0.0)

    threshold, net, n, cost = best
    y_pred = (y_proba >= threshold).astype(int)
    return ThresholdPolicy(
        threshold=round(threshold, 3),
        n_targeted=n,
        expected_net_saving=round(net, 2),
        campaign_cost=round(cost, 2),
        roi=round((net + cost) / cost, 3) if cost else 0.0,
        offer_cost=float(econ["retention_offer_cost"]),
        offer_success_rate=float(econ["offer_success_rate"]),
        horizon_months=int(econ["horizon_months"]),
        precision_at_threshold=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall_at_threshold=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    )


def risk_band(proba: float) -> str:
    if proba >= 0.70:
        return "Critical"
    if proba >= 0.50:
        return "High"
    if proba >= 0.30:
        return "Medium"
    return "Low"
