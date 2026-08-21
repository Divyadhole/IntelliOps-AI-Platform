"""Module 2 tests: preprocessing hygiene, metrics, and the expected-value policy."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from intelliops.churn_model.evaluate import (
    choose_business_threshold,
    classification_report_dict,
    expected_value_of_offer,
    lift_table,
    risk_band,
)
from intelliops.churn_model.features import build_preprocessor, prepare_xy, split_feature_types


@pytest.fixture(scope="module")
def fitted(features, cfg):
    X, y = prepare_xy(features, cfg)
    numeric, categorical = split_feature_types(features, cfg)
    pre = build_preprocessor(numeric, categorical)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    Xt_train, Xt_test = pre.fit_transform(X_train), pre.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt_train, y_train)
    return {"model": model, "pre": pre, "Xt_test": Xt_test, "y_test": y_test,
            "proba": model.predict_proba(Xt_test)[:, 1], "X_test": X_test}


class TestFeatureMatrix:
    def test_target_and_id_never_enter_the_matrix(self, features, cfg):
        X, _ = prepare_xy(features, cfg)
        assert "Churn" not in X.columns
        assert "customerID" not in X.columns

    def test_dollar_reporting_columns_are_excluded_as_leakage(self, features, cfg):
        # annual_margin_at_risk is a deterministic function of MonthlyCharges used
        # only for reporting; leaving it in inflates importance without adding signal.
        X, _ = prepare_xy(features, cfg)
        assert "annual_margin_at_risk" not in X.columns
        assert "annual_revenue_at_risk" not in X.columns

    def test_unseen_category_does_not_crash_the_encoder(self, features, cfg):
        X, _ = prepare_xy(features, cfg)
        numeric, categorical = split_feature_types(features, cfg)
        pre = build_preprocessor(numeric, categorical).fit(X)
        novel = X.head(1).copy()
        novel.loc[novel.index[0], "Contract"] = "Lifetime membership"
        assert pre.transform(novel).shape[1] == pre.transform(X.head(1)).shape[1]


class TestMetrics:
    def test_model_beats_random(self, fitted):
        metrics = classification_report_dict(fitted["y_test"], fitted["proba"])
        assert metrics["roc_auc"] > 0.70
        assert 0.0 <= metrics["brier"] <= 0.25

    def test_all_expected_metrics_present(self, fitted):
        metrics = classification_report_dict(fitted["y_test"], fitted["proba"])
        assert set(metrics) == {"roc_auc", "pr_auc", "log_loss", "brier",
                                "accuracy", "precision", "recall", "f1"}

    def test_top_decile_lifts_above_base_rate(self, fitted):
        table = lift_table(fitted["y_test"], fitted["proba"])
        assert len(table) == 10
        assert table[0]["lift"] > 1.0
        assert table[-1]["cumulative_capture"] == pytest.approx(1.0, abs=1e-6)


class TestBusinessPolicy:
    def test_expected_value_formula(self, cfg):
        econ = cfg["churn_model.economics"]
        ev = expected_value_of_offer([1.0], [100.0], cfg)[0]
        expected = (
            1.0 * econ["offer_success_rate"] * 100.0
            * econ["monthly_margin_multiplier"] * econ["horizon_months"]
            - econ["retention_offer_cost"]
        )
        assert ev == pytest.approx(expected)

    def test_low_margin_customers_are_not_worth_contacting(self, cfg):
        # certain to churn, but pays so little that the offer costs more than it saves
        assert expected_value_of_offer([0.99], [5.0], cfg)[0] < 0

    def test_threshold_is_chosen_not_defaulted(self, fitted, cfg):
        charges = fitted["X_test"]["MonthlyCharges"].to_numpy()
        policy = choose_business_threshold(fitted["y_test"], fitted["proba"], charges, cfg)
        assert 0.0 < policy.threshold < 1.0
        assert policy.n_targeted > 0
        assert policy.expected_net_saving > 0
        assert policy.roi > 1.0

    def test_policy_beats_targeting_everyone(self, fitted, cfg):
        charges = fitted["X_test"]["MonthlyCharges"].to_numpy()
        policy = choose_business_threshold(fitted["y_test"], fitted["proba"], charges, cfg)
        contact_all = float(expected_value_of_offer(fitted["proba"], charges, cfg).sum())
        assert policy.expected_net_saving >= contact_all

    @pytest.mark.parametrize(
        ("proba", "expected"),
        [(0.05, "Low"), (0.35, "Medium"), (0.55, "High"), (0.95, "Critical")],
    )
    def test_risk_bands(self, proba, expected):
        assert risk_band(proba) == expected


class TestCalibrationChoice:
    def test_probabilities_stay_inside_the_open_unit_interval(self, fitted):
        # Saturated 0.0/1.0 probabilities propagate false certainty into every
        # downstream dollar figure; the platform must never emit them.
        proba = fitted["proba"]
        assert np.all(proba > 0.0) and np.all(proba < 1.0)


class TestTrackingIsOptional:
    """Experiment tracking must never be able to fail a training run.

    Regression test for a real break: mlflow 3.15 started raising on the old
    ``file:./mlruns`` store, which turned a logging backend into a build-breaking
    dependency and took CI's end-to-end job down with it. Unit tests passed the
    whole time, because they never touched mlflow — hence this test.
    """

    def test_relative_sqlite_uri_is_resolved_absolute(self, cfg):
        uri = cfg.mlflow_uri
        assert uri.startswith("sqlite:////") or not uri.startswith("sqlite:///"), (
            "a relative tracking URI silently creates a second database when the run "
            "starts from a different working directory"
        )

    def test_training_continues_when_the_tracking_backend_refuses(self, cfg, monkeypatch):
        from intelliops.churn_model.train import _mlflow_run

        try:
            import mlflow
        except ImportError:
            run_ctx, tracker = _mlflow_run(cfg)
            assert tracker is None
            with run_ctx:
                pass
            return

        def explode(*_args, **_kwargs):
            raise RuntimeError("The filesystem tracking backend is in maintenance mode")

        monkeypatch.setattr(mlflow, "set_tracking_uri", explode)
        run_ctx, tracker = _mlflow_run(cfg)
        assert tracker is None, "a refusing backend must disable tracking, not raise"
        with run_ctx:  # the no-op context still has to be usable
            pass

    def test_tracking_can_be_switched_off_entirely(self, cfg, monkeypatch):
        from intelliops.churn_model.train import _mlflow_run

        monkeypatch.setitem(cfg._data["mlflow"], "enabled", False)
        run_ctx, tracker = _mlflow_run(cfg)
        assert tracker is None
        with run_ctx:
            pass
