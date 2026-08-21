"""Shared fixtures. Tests run on a small synthetic slice and never touch the warehouse."""

from __future__ import annotations

import pandas as pd
import pytest

from intelliops.config import load_config
from intelliops.data_pipeline.synthetic import generate_customers, generate_reviews
from intelliops.data_pipeline.transform import clean_customers, engineer_features


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_customers() -> pd.DataFrame:
    return generate_customers(n=600, seed=7)


@pytest.fixture(scope="session")
def features(raw_customers, cfg) -> pd.DataFrame:
    return engineer_features(clean_customers(raw_customers, cfg), cfg)


@pytest.fixture(scope="session")
def reviews(raw_customers) -> pd.DataFrame:
    return generate_reviews(raw_customers["customerID"].tolist(), n=400, seed=7)
