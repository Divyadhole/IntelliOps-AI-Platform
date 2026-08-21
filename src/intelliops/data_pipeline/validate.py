"""Data-contract validation — the gate between raw data and the warehouse.

Implements the checks a Great Expectations / dbt-test suite would enforce, but
dependency-free so the repo stays runnable: schema presence, null budgets,
duplicate budgets, range assertions and target-class sanity. Results are returned
as a tidy DataFrame that is itself persisted to ``ops_data_quality`` so quality is
tracked over time rather than printed once and forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(RuntimeError):
    """Raised when a blocking data-quality expectation fails."""


@dataclass
class CheckResult:
    check: str
    column: str
    status: str          # PASS | WARN | FAIL
    observed: float
    threshold: float | None
    detail: str = ""


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "FAIL"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])

    def summary(self) -> str:
        counts = pd.Series([r.status for r in self.results]).value_counts().to_dict()
        return " | ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def validate_customers(df: pd.DataFrame, cfg: Config | None = None) -> ValidationReport:
    """Run the full customer data contract and return a structured report."""
    cfg = cfg or load_config()
    report = ValidationReport()

    # 1. schema ------------------------------------------------------------
    for col in cfg["validation.required_columns"]:
        present = col in df.columns
        report.add(
            CheckResult(
                check="column_exists",
                column=col,
                status="PASS" if present else "FAIL",
                observed=float(present),
                threshold=1.0,
                detail="" if present else "required column missing from source",
            )
        )

    # 2. null budget -------------------------------------------------------
    max_null = float(cfg["validation.max_null_fraction"])
    for col in df.columns:
        frac = float(df[col].isna().mean())
        report.add(
            CheckResult(
                check="null_fraction",
                column=col,
                status="PASS" if frac <= max_null else "WARN",
                observed=round(frac, 5),
                threshold=max_null,
            )
        )

    # 3. duplicates --------------------------------------------------------
    dup_frac = float(df.duplicated().mean())
    max_dup = float(cfg["validation.max_duplicate_fraction"])
    report.add(
        CheckResult(
            check="duplicate_fraction",
            column="<row>",
            status="PASS" if dup_frac <= max_dup else "WARN",
            observed=round(dup_frac, 5),
            threshold=max_dup,
            detail=f"{int(df.duplicated().sum())} exact duplicate rows (deduplicated downstream)",
        )
    )

    # 4. numeric ranges ----------------------------------------------------
    for col, (lo, hi) in (cfg.get("validation.ranges") or {}).items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = float(((series < lo) | (series > hi)).mean())
        report.add(
            CheckResult(
                check="value_in_range",
                column=col,
                status="PASS" if out_of_range == 0 else "FAIL",
                observed=round(out_of_range, 5),
                threshold=0.0,
                detail=f"expected [{lo}, {hi}]",
            )
        )

    # 5. target sanity -----------------------------------------------------
    target = cfg.get("churn_model.target", "Churn")
    if target in df.columns:
        rate = float((df[target].astype(str).str.strip().str.lower() == "yes").mean())
        healthy = 0.02 <= rate <= 0.80
        report.add(
            CheckResult(
                check="target_base_rate",
                column=target,
                status="PASS" if healthy else "FAIL",
                observed=round(rate, 5),
                threshold=None,
                detail="churn base rate outside plausible range" if not healthy else "",
            )
        )

    logger.info("Validation complete: %s", report.summary())
    for failure in report.failures:
        logger.error("FAILED CHECK %s on %s: observed=%s %s",
                     failure.check, failure.column, failure.observed, failure.detail)

    if report.failures and bool(cfg.get("validation.fail_on_error", True)):
        raise DataValidationError(
            f"{len(report.failures)} blocking data-quality check(s) failed: "
            + ", ".join(f"{f.check}:{f.column}" for f in report.failures)
        )
    return report
