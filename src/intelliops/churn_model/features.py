"""Feature preprocessing for the churn model.

Everything lives inside a single sklearn ``Pipeline`` so that training and serving
apply *byte-identical* transformations — the most common source of silent
train/serve skew is a preprocessing step that exists only in the notebook.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import Config, load_config

# Columns that leak the outcome or carry no signal, excluded from every model.
LEAKY_OR_ID_COLUMNS = {
    "customerID", "Churn",
    "annual_revenue_at_risk", "annual_margin_at_risk",  # deterministic functions of charges, used for $ reporting only
    "clv_to_date",                                       # duplicate of TotalCharges
    "last_review_date",
}


def split_feature_types(df: pd.DataFrame, cfg: Config | None = None) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the model matrix."""
    cfg = cfg or load_config()
    drop = set(LEAKY_OR_ID_COLUMNS) | {cfg.get("churn_model.target", "Churn"),
                                       cfg.get("churn_model.id_column", "customerID")}
    usable = [c for c in df.columns if c not in drop]
    numeric = [c for c in usable if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in usable if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot encode categoricals."""
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def expanded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Post-encoding feature names, used to label SHAP values."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - older sklearn fallback
        return [f"f{i}" for i in range(preprocessor.transform_shape_[1])]


def prepare_xy(df: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cfg = cfg or load_config()
    target = cfg.get("churn_model.target", "Churn")
    raw = df[target]
    if pd.api.types.is_numeric_dtype(raw):
        y = raw.astype(int)
    else:  # defensive: a warehouse round-trip can hand back "Yes"/"No" strings
        y = raw.astype("string").str.strip().str.lower().isin(["yes", "y", "true", "1"]).astype(int)
    numeric, categorical = split_feature_types(df, cfg)
    X = df[numeric + categorical].copy()
    return X, y
