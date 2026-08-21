"""Module 4 tests: the serving contract.

These run in both states of the world — with a trained model on disk and without —
because CI lints and tests a fresh checkout before the end-to-end job has produced
any artifacts. A service that must be fully provisioned before its contract can be
tested is a service that cannot be tested in CI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intelliops.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def model_ready(client) -> bool:
    return bool(client.get("/health").json()["model_loaded"])


VALID_CUSTOMER = {
    "customerID": "TEST-0001",
    "tenure": 2,
    "MonthlyCharges": 98.5,
    "TotalCharges": 197.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "TechSupport": "No",
}


class TestOperations:
    def test_health_always_answers(self, client):
        body = client.get("/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert isinstance(body["model_loaded"], bool)

    def test_health_explains_why_it_is_degraded(self, client):
        body = client.get("/health").json()
        if body["status"] == "degraded":
            assert body["detail"]["errors"], "a degraded service must say what is missing"

    def test_openapi_schema_is_valid(self, client):
        schema = client.get("/openapi.json").json()
        assert "/predict_churn" in schema["paths"]
        assert "/ask" in schema["paths"]

    def test_every_response_is_traceable(self, client):
        response = client.get("/health")
        assert response.headers["x-request-id"]
        assert float(response.headers["x-process-time-ms"]) >= 0

    def test_metrics_expose_latency_percentiles(self, client):
        client.get("/health")
        body = client.get("/metrics").json()
        assert body["counters"]
        assert body["latency"]["/health"]["p95_ms"] >= 0


class TestValidation:
    def test_missing_required_field_is_rejected(self, client):
        assert client.post("/predict_churn", json={"tenure": 5}).status_code == 422

    def test_negative_tenure_is_rejected(self, client):
        payload = VALID_CUSTOMER | {"tenure": -3}
        assert client.post("/predict_churn", json=payload).status_code == 422

    def test_unknown_fields_are_accepted(self, client, model_ready):
        if not model_ready:
            pytest.skip("model not trained in this environment")
        payload = VALID_CUSTOMER | {"SomeNewCrmField": "whatever"}
        assert client.post("/predict_churn", json=payload).status_code == 200

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/predict_churn/batch", json={"customers": []}).status_code == 422

    def test_too_short_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": "a"}).status_code == 422


class TestScoringContract:
    def test_missing_model_returns_503_not_500(self, client, model_ready):
        if model_ready:
            pytest.skip("model is loaded here; the 503 path is covered on a clean checkout")
        response = client.post("/predict_churn", json=VALID_CUSTOMER)
        assert response.status_code == 503
        assert "make train" in response.json()["detail"]

    def test_prediction_shape(self, client, model_ready):
        if not model_ready:
            pytest.skip("model not trained in this environment")
        body = client.post("/predict_churn", json=VALID_CUSTOMER).json()
        assert 0.0 < body["churn_probability"] < 1.0
        assert body["risk_band"] in {"Low", "Medium", "High", "Critical"}
        assert body["customer_id"] == "TEST-0001"
        assert body["top_drivers"], "an explanation is part of the contract, not an extra"

    def test_explanations_are_actionable(self, client, model_ready):
        if not model_ready:
            pytest.skip("model not trained in this environment")
        drivers = client.post("/predict_churn", json=VALID_CUSTOMER).json()["top_drivers"]
        for driver in drivers:
            assert driver["reason"] and not driver["reason"].startswith("f")
            assert driver["direction"] in {"increases risk", "reduces risk"}

    def test_high_and_low_risk_profiles_are_separated(self, client, model_ready):
        if not model_ready:
            pytest.skip("model not trained in this environment")
        loyal = VALID_CUSTOMER | {"customerID": "LOYAL", "tenure": 60, "MonthlyCharges": 40.0,
                                  "TotalCharges": 2400.0, "Contract": "Two year",
                                  "PaymentMethod": "Credit card (automatic)", "TechSupport": "Yes"}
        risky = client.post("/predict_churn", json=VALID_CUSTOMER).json()["churn_probability"]
        safe = client.post("/predict_churn", json=loyal).json()["churn_probability"]
        assert risky > safe + 0.20

    def test_batch_returns_one_prediction_per_customer(self, client, model_ready):
        if not model_ready:
            pytest.skip("model not trained in this environment")
        payload = {"customers": [
            {"tenure": 2, "MonthlyCharges": 98.5, "Contract": "Month-to-month"},
            {"tenure": 2, "MonthlyCharges": 98.5, "Contract": "Month-to-month"},
            {"tenure": 55, "MonthlyCharges": 30.0, "Contract": "Two year"},
        ]}
        body = client.post("/predict_churn/batch", json=payload).json()
        assert body["count"] == 3
        assert len(body["predictions"]) == 3


class TestIntelligenceContract:
    def test_answer_is_grounded_in_citations(self, client):
        response = client.post("/ask", json={"question": "Why are customers leaving?"})
        if response.status_code == 503:
            pytest.skip("RAG index not built in this environment")
        body = response.json()
        assert body["answer"].strip()
        assert body["citations"], "an ungrounded answer is not an acceptable response"
        assert body["generated_by"] in {"llm", "deterministic"}
