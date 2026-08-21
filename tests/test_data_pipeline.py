"""Module 1 tests: the data contract, the cleaning rules and the feature definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelliops.data_pipeline.run_pipeline import run
from intelliops.data_pipeline.transform import clean_customers, engineer_features, prepare_for_inference
from intelliops.data_pipeline.validate import DataValidationError, validate_customers


class TestSyntheticSource:
    def test_has_the_telco_schema(self, raw_customers, cfg):
        for col in cfg["validation.required_columns"]:
            assert col in raw_customers.columns

    def test_reproduces_the_real_datasets_messiness(self, raw_customers):
        # blank TotalCharges strings and duplicate rows are the two defects that
        # break naive Telco notebooks; the generator must emit both.
        assert (raw_customers["TotalCharges"].astype(str).str.strip() == "").any()
        assert raw_customers.duplicated().any()

    def test_churn_is_learnable_not_random(self, raw_customers):
        rate = (raw_customers["Churn"] == "Yes").mean()
        assert 0.10 < rate < 0.80
        mtm = raw_customers[raw_customers["Contract"] == "Month-to-month"]["Churn"].eq("Yes").mean()
        two_year = raw_customers[raw_customers["Contract"] == "Two year"]["Churn"].eq("Yes").mean()
        assert mtm > two_year, "month-to-month customers must churn more, or the signal is absent"


class TestValidation:
    def test_clean_data_passes(self, raw_customers, cfg):
        report = validate_customers(raw_customers, cfg)
        assert not report.failures
        assert len(report.to_frame()) > 0

    def test_missing_required_column_is_blocking(self, raw_customers, cfg):
        broken = raw_customers.drop(columns=["Contract"])
        with pytest.raises(DataValidationError):
            validate_customers(broken, cfg)

    def test_out_of_range_values_are_blocking(self, raw_customers, cfg):
        broken = raw_customers.copy()
        broken.loc[broken.index[0], "tenure"] = 9_999
        with pytest.raises(DataValidationError):
            validate_customers(broken, cfg)

    def test_duplicates_warn_but_do_not_block(self, raw_customers, cfg):
        report = validate_customers(raw_customers, cfg)
        dup = [r for r in report.results if r.check == "duplicate_fraction"][0]
        assert dup.status in {"PASS", "WARN"}


class TestCleaning:
    def test_blank_total_charges_are_repaired(self, raw_customers, cfg):
        cleaned = clean_customers(raw_customers, cfg)
        assert cleaned["TotalCharges"].notna().all()
        assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])

    def test_target_is_binary(self, raw_customers, cfg):
        cleaned = clean_customers(raw_customers, cfg)
        assert set(cleaned["Churn"].unique()) <= {0, 1}

    def test_duplicates_removed(self, raw_customers, cfg):
        cleaned = clean_customers(raw_customers, cfg)
        assert not cleaned.duplicated().any()
        assert cleaned["customerID"].is_unique


class TestFeatureEngineering:
    EXPECTED = ["avg_charge_per_month", "charge_drift", "services_count",
                "tenure_bucket", "is_month_to_month", "annual_margin_at_risk"]

    def test_expected_features_exist(self, features):
        for col in self.EXPECTED:
            assert col in features.columns

    def test_no_nulls_introduced(self, features):
        assert features[self.EXPECTED].isna().sum().sum() == 0

    def test_avg_charge_is_consistent_with_billing(self, features):
        tenured = features[features["tenure"] > 0]
        expected = tenured["TotalCharges"] / tenured["tenure"]
        assert np.allclose(tenured["avg_charge_per_month"], expected)

    def test_zero_tenure_falls_back_to_monthly_charge(self, features):
        new = features[features["tenure"] == 0]
        if len(new):
            assert np.allclose(new["avg_charge_per_month"], new["MonthlyCharges"])


class TestServingParity:
    """The serving transform must match training *and* preserve row count."""

    def test_inference_transform_preserves_every_row(self, cfg):
        # two customers with no id at all — deduplication here would silently
        # return one prediction for a two-customer request
        batch = pd.DataFrame([
            {"tenure": 2, "MonthlyCharges": 90.0, "Contract": "Month-to-month"},
            {"tenure": 2, "MonthlyCharges": 90.0, "Contract": "Month-to-month"},
        ])
        assert len(prepare_for_inference(batch, cfg)) == 2

    def test_inference_and_training_produce_the_same_columns(self, raw_customers, cfg):
        training = engineer_features(clean_customers(raw_customers, cfg), cfg)
        serving = prepare_for_inference(raw_customers.head(20), cfg)
        assert set(training.columns) == set(serving.columns)


def test_pipeline_runs_end_to_end_without_persisting(cfg):
    result = run(cfg, persist=False)
    assert result.summary()["customers"] > 0
    assert result.features.shape[1] > result.customers.shape[1]
    assert len(result.quality) > 0
