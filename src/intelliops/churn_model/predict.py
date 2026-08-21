"""Inference wrapper shared by the API, the dashboard and the batch scorer.

One class, one code path. If scoring logic existed separately in the API and in the
batch job they would drift within a month; this is the object both import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger
from .evaluate import expected_value_of_offer, risk_band
from .explain import explain_customer

logger = get_logger(__name__)


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the API starts before ``make train`` has produced a model."""


class ChurnScorer:
    """Loads the trained bundle and turns raw customer rows into decisions."""

    def __init__(self, bundle: dict[str, Any] | None = None, cfg: Config | None = None,
                 model_path: str | Path | None = None) -> None:
        self.cfg = cfg or load_config()
        if bundle is None:
            path = Path(model_path) if model_path else self.cfg.path("api.model_path")
            if not path.exists():
                raise ModelNotTrainedError(
                    f"No model at {path}. Run `make train` (or `python -m intelliops.churn_model.train`) first."
                )
            bundle = joblib.load(path)
            logger.info("Loaded model bundle: %s trained %s", bundle.get("model_name"), bundle.get("trained_at"))
        self.bundle = bundle
        self.preprocessor = bundle["preprocessor"]
        self.model = bundle["model"]
        self.input_columns: list[str] = bundle["input_columns"]
        self.threshold: float = float(bundle.get("threshold", 0.5))

    # ------------------------------------------------------------- metadata
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.bundle.get("model_name"),
            "trained_at": self.bundle.get("trained_at"),
            "threshold": self.threshold,
            "metrics": self.bundle.get("metrics", {}),
            "n_features": len(self.bundle.get("feature_names", [])),
        }

    # -------------------------------------------------------------- helpers
    def _ensure_engineered(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply training-time feature engineering to raw request payloads.

        The API accepts *raw* customer attributes (tenure, Contract, MonthlyCharges…),
        but the model was trained on engineered features (is_month_to_month,
        charge_drift, services_count…). Without this step those columns arrive as
        NaN and get median-imputed — the request silently scores against the average
        customer on exactly the fields that carry the most signal. This is the
        train/serve skew that makes production models quietly worse than their
        offline metrics; running the same transform on both sides is the fix.
        """
        missing = [c for c in self.input_columns if c not in df.columns]
        if not missing:
            return df
        try:
            from ..data_pipeline.transform import prepare_for_inference

            prepared = prepare_for_inference(df, self.cfg)
            still_missing = [c for c in self.input_columns if c not in prepared.columns]
            if still_missing:
                logger.debug("Imputing %d columns unavailable at request time: %s",
                             len(still_missing), still_missing[:8])
            return prepared
        except Exception as exc:
            logger.warning("Could not derive engineered features (%s) — falling back to imputation", exc)
            return df

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer, then reindex to the training schema; anything still absent is imputed."""
        return self._ensure_engineered(df).reindex(columns=self.input_columns)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(self._align(df))

    # ------------------------------------------------------------- scoring
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.encode(df))[:, 1]

    def score_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Batch-score a feature table into the decision schema written to the warehouse."""
        proba = self.predict_proba(features)
        monthly = features["MonthlyCharges"].to_numpy() if "MonthlyCharges" in features else np.zeros(len(features))
        ev = expected_value_of_offer(proba, monthly, self.cfg)
        econ = self.cfg["churn_model.economics"]
        margin_at_risk = monthly * float(econ["monthly_margin_multiplier"]) * float(econ["horizon_months"])

        # Callers may omit customerID (ad-hoc scoring). Fall back to a positional id
        # rather than emitting NaN, which would break every downstream string contract.
        positional = pd.Series([f"row-{i}" for i in range(len(features))])
        if "customerID" in features.columns:
            ids = features["customerID"].astype("string").reset_index(drop=True).fillna(positional)
        else:
            ids = positional

        out = pd.DataFrame(
            {
                "customerID": ids.astype(str),
                "churn_probability": proba,
                "risk_band": [risk_band(p) for p in proba],
                "expected_value_of_offer": np.round(ev, 2),
                "margin_at_risk": np.round(margin_at_risk, 2),
                "targeted_by_policy": ((proba >= self.threshold) & (ev > 0)).astype(int),
                "scored_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )
        out["recommended_action"] = np.where(
            out["targeted_by_policy"] == 1,
            np.where(out["churn_probability"] >= 0.70,
                     "Priority save call + retention offer",
                     "Retention offer / contract upgrade"),
            np.where(out["churn_probability"] >= self.threshold,
                     "Monitor — offer not economic at this margin",
                     "No action"),
        )
        logger.info("Scored %d customers | %d targeted by policy | expected net saving $%.0f",
                    len(out), int(out["targeted_by_policy"].sum()),
                    float(ev[(proba >= self.threshold) & (ev > 0)].sum()))
        return out

    # ------------------------------------------------------- single record
    def explain_one(self, record: dict[str, Any], top_n: int = 4) -> dict[str, Any]:
        """Score one customer and return probability, band, economics and SHAP drivers."""
        df = pd.DataFrame([record])
        encoded = self.encode(df)
        proba = float(self.model.predict_proba(encoded)[0, 1])
        monthly = float(record.get("MonthlyCharges", 0.0) or 0.0)
        ev = float(expected_value_of_offer([proba], [monthly], self.cfg)[0])
        drivers = explain_customer(self.bundle, encoded[0], top_n=top_n, cfg=self.cfg)
        targeted = proba >= self.threshold and ev > 0
        return {
            "customer_id": record.get("customerID"),
            "churn_probability": round(proba, 4),
            "risk_band": risk_band(proba),
            "decision_threshold": self.threshold,
            "expected_value_of_offer": round(ev, 2),
            "targeted_by_policy": bool(targeted),
            "recommended_action": (
                "Priority save call + retention offer" if targeted and proba >= 0.70
                else "Retention offer / contract upgrade" if targeted
                else "Monitor — offer not economic at this margin" if proba >= self.threshold
                else "No action"
            ),
            "top_drivers": drivers,
            "model": self.bundle.get("model_name"),
        }
