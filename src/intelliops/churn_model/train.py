"""Churn model training: baseline → gradient boosting → calibration → selection.

Design decisions worth defending in an interview:

* **Preprocessor is fitted once and stored separately** from the classifier. SHAP
  needs the *encoded* matrix and the *raw* tree model; wrapping everything in one
  opaque estimator makes explainability awkward, so the bundle keeps both.
* **Probabilities are calibrated** (Platt/sigmoid by default). An uncalibrated
  XGBoost score of 0.87 is not an 87% chance of churn, and every downstream dollar
  figure multiplies by that probability. Isotonic scored marginally better on Brier
  but saturates to exactly 0.0 and 1.0 at the tails — false certainty that then
  propagates into the expected-value calculation — so sigmoid is the default.
* **Selection is by cross-validated ROC-AUC, but the decision threshold is chosen
  by expected value**, not by 0.5 or by F1. Retention offers cost money; the model
  should only fire when the expected saved margin exceeds the offer cost.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ..config import Config, load_config
from ..data_pipeline import warehouse
from ..logging_utils import get_logger, stage
from .evaluate import choose_business_threshold, classification_report_dict, lift_table
from .features import build_preprocessor, expanded_feature_names, prepare_xy, split_feature_types

logger = get_logger(__name__)


@dataclass
class ModelResult:
    name: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    test_metrics: dict[str, float]


def _build_candidates(cfg: Config) -> dict[str, Any]:
    """Instantiate every enabled model from config. Missing optional deps are skipped."""
    spec = cfg["churn_model.models"]
    seed = cfg.seed
    candidates: dict[str, Any] = {}

    if spec.get("logistic_regression", {}).get("enabled"):
        candidates["logistic_regression"] = LogisticRegression(
            random_state=seed, **spec["logistic_regression"]["params"]
        )
    if spec.get("random_forest", {}).get("enabled"):
        candidates["random_forest"] = RandomForestClassifier(
            random_state=seed, n_jobs=-1, **spec["random_forest"]["params"]
        )
    if spec.get("xgboost", {}).get("enabled"):
        try:
            from xgboost import XGBClassifier

            candidates["xgboost"] = XGBClassifier(
                random_state=seed, n_jobs=-1, tree_method="hist", **spec["xgboost"]["params"]
            )
        except ImportError:
            logger.warning("xgboost not installed — skipping that candidate")
    if spec.get("lightgbm", {}).get("enabled"):
        try:
            from lightgbm import LGBMClassifier

            candidates["lightgbm"] = LGBMClassifier(
                random_state=seed, n_jobs=-1, **spec["lightgbm"]["params"]
            )
        except ImportError:
            logger.warning("lightgbm not installed — skipping that candidate")

    logger.info("Candidate models: %s", ", ".join(candidates))
    return candidates


def _mlflow_run(cfg: Config):
    """Return an MLflow run context, or a no-op if MLflow is unavailable/disabled."""
    from contextlib import nullcontext

    if not cfg.get("mlflow.enabled", True):
        return nullcontext(), None
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — experiment tracking disabled for this run")
        return nullcontext(), None
    mlflow.set_tracking_uri(cfg["mlflow.tracking_uri"])
    mlflow.set_experiment(cfg["mlflow.experiment"])
    return mlflow.start_run(run_name=f"churn-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"), mlflow


def train(cfg: Config | None = None, features: pd.DataFrame | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    seed = cfg.seed

    # ---------------------------------------------------------------- data
    with stage(logger, "load feature table"):
        if features is None:
            features = warehouse.read_table(cfg["warehouse.schema_tables.features"], cfg)
        logger.info("Feature table: %d rows × %d cols", len(features), features.shape[1])

    X, y = prepare_xy(features, cfg)
    numeric, categorical = split_feature_types(features, cfg)
    logger.info("Model matrix: %d numeric, %d categorical | churn base rate %.3f",
                len(numeric), len(categorical), y.mean())

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)),
        test_size=float(cfg["churn_model.test_size"]),
        stratify=y, random_state=seed,
    )

    # -------------------------------------------------------- preprocessing
    with stage(logger, "fit preprocessor"):
        preprocessor = build_preprocessor(numeric, categorical)
        Xt_train = preprocessor.fit_transform(X_train)
        Xt_test = preprocessor.transform(X_test)
        feature_names = expanded_feature_names(preprocessor)
        logger.info("Encoded matrix: %s → %d features", Xt_train.shape, len(feature_names))

    # ------------------------------------------------------------ training
    cv = StratifiedKFold(n_splits=int(cfg["churn_model.cv_folds"]), shuffle=True, random_state=seed)
    results: list[ModelResult] = []
    fitted: dict[str, Any] = {}

    run_ctx, mlflow = _mlflow_run(cfg)
    with run_ctx:
        for name, model in _build_candidates(cfg).items():
            with stage(logger, f"train {name}"):
                cv_scores = cross_val_score(model, Xt_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
                model.fit(Xt_train, y_train)
                proba = model.predict_proba(Xt_test)[:, 1]
                metrics = classification_report_dict(y_test, proba)
                results.append(ModelResult(name, float(cv_scores.mean()), float(cv_scores.std()), metrics))
                fitted[name] = model
                logger.info("%-20s CV ROC-AUC %.4f ±%.4f | test ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f",
                            name, cv_scores.mean(), cv_scores.std(),
                            metrics["roc_auc"], metrics["pr_auc"], metrics["brier"])
                if mlflow:
                    mlflow.log_params({f"{name}__{k}": v for k, v in model.get_params().items()
                                       if isinstance(v, (int, float, str, bool)) and v is not None})
                    mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
                    mlflow.log_metric(f"{name}_cv_roc_auc", float(cv_scores.mean()))

        # ------------------------------------------------------- selection
        best = max(results, key=lambda r: r.cv_roc_auc_mean)
        logger.info("Selected model: %s (CV ROC-AUC %.4f)", best.name, best.cv_roc_auc_mean)
        base_model = fitted[best.name]

        # ----------------------------------------------------- calibration
        method = cfg.get("churn_model.calibration")
        if method:
            with stage(logger, f"calibrate ({method})"):
                calibrated = CalibratedClassifierCV(base_model, method=method, cv=3)
                calibrated.fit(Xt_train, y_train)
                cal_proba = calibrated.predict_proba(Xt_test)[:, 1]
                cal_metrics = classification_report_dict(y_test, cal_proba)
                logger.info("Calibration: Brier %.4f → %.4f | ROC-AUC %.4f → %.4f",
                            best.test_metrics["brier"], cal_metrics["brier"],
                            best.test_metrics["roc_auc"], cal_metrics["roc_auc"])
                scorer, test_proba, final_metrics = calibrated, cal_proba, cal_metrics
        else:
            scorer, test_proba, final_metrics = base_model, base_model.predict_proba(Xt_test)[:, 1], best.test_metrics

        # ------------------------------------------- business decision layer
        with stage(logger, "optimise decision threshold on expected value"):
            monthly_charges_test = features.iloc[idx_test]["MonthlyCharges"].to_numpy()
            policy = choose_business_threshold(y_test.to_numpy(), test_proba, monthly_charges_test, cfg)
            logger.info("Policy: threshold=%.3f | targets %d/%d customers | expected net saving $%.0f "
                        "| campaign ROI %.2fx",
                        policy.threshold, policy.n_targeted, len(test_proba),
                        policy.expected_net_saving, policy.roi)

        if mlflow:
            mlflow.log_param("selected_model", best.name)
            mlflow.log_param("calibration", method or "none")
            mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
            mlflow.log_metrics({"decision_threshold": policy.threshold,
                                "expected_net_saving": policy.expected_net_saving,
                                "campaign_roi": policy.roi})

    # -------------------------------------------------------------- persist
    bundle = {
        "preprocessor": preprocessor,
        "model": scorer,
        "base_model": base_model,
        "model_name": best.name,
        "feature_names": feature_names,
        # Reference distribution for SHAP. Without a background sample, per-request
        # explanations are computed against the request itself and come out all-zero.
        "background": np.asarray(Xt_train)[
            np.random.default_rng(seed).choice(len(Xt_train), size=min(200, len(Xt_train)), replace=False)
        ],
        "input_columns": list(X.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "threshold": policy.threshold,
        "metrics": final_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "economics": cfg["churn_model.economics"],
    }
    model_path = cfg.path("api.model_path")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    logger.info("Saved model bundle → %s", model_path)

    report = {
        "trained_at": bundle["trained_at"],
        "selected_model": best.name,
        "calibration": method or "none",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_rate": float(y.mean()),
        "candidates": [asdict(r) for r in results],
        "final_metrics": final_metrics,
        "decision_policy": asdict(policy),
        "lift_table": lift_table(y_test.to_numpy(), test_proba),
    }
    report_path = cfg.path("paths.reports") / "model_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation report → %s", report_path)

    # ------------------------------------- score the full base for the app
    with stage(logger, "score full customer base and write predictions"):
        from .predict import ChurnScorer

        scorer_obj = ChurnScorer(bundle=bundle, cfg=cfg)
        predictions = scorer_obj.score_frame(features)
        warehouse.write_table(predictions, cfg["warehouse.schema_tables.predictions"], cfg)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IntelliOps churn model")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    train(load_config(args.config) if args.config else None)


if __name__ == "__main__":
    main()
