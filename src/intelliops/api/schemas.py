"""Pydantic request/response contracts for the serving layer.

The customer payload is deliberately permissive (``extra="allow"``): the scorer
reindexes to the training schema and imputes what is missing, so a caller adding a
new CRM field does not take the endpoint down. What is *not* permissive is the
response — every field a consumer depends on is typed and documented.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerPayload(BaseModel):
    """Raw customer attributes. Unknown extra fields are accepted and ignored."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "customerID": "0001-ABCDE", "tenure": 3, "MonthlyCharges": 95.4, "TotalCharges": 286.2,
                "Contract": "Month-to-month", "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check", "TechSupport": "No", "OnlineSecurity": "No",
            }
        },
    )

    customerID: str | None = Field(default=None, description="Customer identifier, echoed back")
    tenure: float = Field(..., ge=0, le=200, description="Months since acquisition")
    MonthlyCharges: float = Field(..., ge=0, description="Current monthly charge, USD")
    TotalCharges: float | None = Field(default=None, ge=0, description="Lifetime billed amount, USD")
    Contract: str | None = Field(default=None, examples=["Month-to-month", "One year", "Two year"])
    InternetService: str | None = Field(default=None, examples=["Fiber optic", "DSL", "No"])
    PaymentMethod: str | None = Field(default=None, examples=["Electronic check", "Credit card (automatic)"])


class Driver(BaseModel):
    feature: str
    contribution: float = Field(..., description="Signed SHAP contribution to the log-odds")
    direction: Literal["increases risk", "reduces risk"]
    reason: str = Field(..., description="Plain-English rendering of the driver")
    recommended_action: str


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    churn_probability: float = Field(..., ge=0, le=1, description="Calibrated probability of churn")
    risk_band: Literal["Low", "Medium", "High", "Critical"]
    decision_threshold: float
    expected_value_of_offer: float = Field(
        ..., description="Expected USD gain from a retention offer; negative means do not contact"
    )
    targeted_by_policy: bool
    recommended_action: str
    top_drivers: list[Driver] = []
    model: str | None = None


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerPayload] = Field(..., min_length=1, max_length=1000)
    include_drivers: bool = False


class BatchPredictionResponse(BaseModel):
    count: int
    targeted: int
    total_expected_value: float
    predictions: list[PredictionResponse]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Why are customers leaving?"])
    top_k: int | None = Field(default=None, ge=1, le=25)
    sources: list[str] | None = Field(
        default=None, description="Restrict evidence to these sources", examples=[["review", "topic_summary"]]
    )


class Citation(BaseModel):
    id: str
    doc_id: str
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = []
    generated_by: Literal["llm", "deterministic"]
    model: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    rag_loaded: bool
    warehouse_reachable: bool
    version: str
    detail: dict[str, Any] = {}
