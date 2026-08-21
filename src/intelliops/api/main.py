"""FastAPI serving layer.

Production behaviours that a demo API usually skips and an interviewer will look for:

* heavy objects (model bundle, vector index) load **once at startup**, not per request;
* the service starts *degraded* rather than crash-looping if the model is missing,
  and ``/health`` says so — an orchestrator can then hold traffic off it;
* every response carries a request id and a server-side latency header;
* an in-process metrics endpoint exposes request counts, error counts and p95 latency.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import load_config
from ..data_pipeline import warehouse
from ..logging_utils import get_logger
from .schemas import (
    AskRequest,
    AskResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerPayload,
    HealthResponse,
    PredictionResponse,
)

logger = get_logger(__name__)

STATE: dict[str, Any] = {"scorer": None, "assistant": None, "errors": {}}
LATENCIES: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
COUNTERS: dict[str, int] = defaultdict(int)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model and the RAG index before the first request arrives."""
    cfg = load_config()
    try:
        from ..churn_model.predict import ChurnScorer

        STATE["scorer"] = ChurnScorer(cfg=cfg)
        logger.info("Churn model ready: %s", STATE["scorer"].metadata)
    except Exception as exc:
        STATE["errors"]["model"] = str(exc)
        logger.error("Model unavailable at startup — /predict_churn will return 503: %s", exc)

    try:
        from ..rag_assistant.assistant import BusinessAnalystAssistant

        STATE["assistant"] = BusinessAnalystAssistant(cfg=cfg)
        logger.info("RAG assistant ready (%d documents)", len(STATE["assistant"].store.documents))
    except Exception as exc:
        STATE["errors"]["rag"] = str(exc)
        logger.error("RAG index unavailable at startup — /ask will return 503: %s", exc)

    yield
    logger.info("Shutting down; served %d requests", sum(COUNTERS.values()))


app = FastAPI(
    title="IntelliOps AI Platform API",
    description=(
        "Churn scoring with SHAP explanations, an expected-value retention policy, "
        "and a retrieval-augmented business analyst over customer evidence."
    ),
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        COUNTERS[f"error:{request.url.path}"] += 1
        logger.exception("Unhandled error on %s [req %s]", request.url.path, request_id)
        return JSONResponse(status_code=500, content={"detail": "Internal error", "request_id": request_id})
    elapsed_ms = (time.perf_counter() - started) * 1000
    LATENCIES[request.url.path].append(elapsed_ms)
    COUNTERS[f"request:{request.url.path}"] += 1
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{elapsed_ms:.2f}"
    return response


def _scorer():
    if STATE["scorer"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Run `make train` first. Cause: {STATE['errors'].get('model')}",
        )
    return STATE["scorer"]


def _assistant():
    if STATE["assistant"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"RAG index not loaded. Run `make rag-index` first. Cause: {STATE['errors'].get('rag')}",
        )
    return STATE["assistant"]


# ---------------------------------------------------------------- operations
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    cfg = load_config()
    warehouse_ok = warehouse.table_exists(cfg["warehouse.schema_tables.features"], cfg)
    model_ok = STATE["scorer"] is not None
    rag_ok = STATE["assistant"] is not None
    return HealthResponse(
        status="ok" if (model_ok and warehouse_ok) else "degraded",
        model_loaded=model_ok,
        rag_loaded=rag_ok,
        warehouse_reachable=warehouse_ok,
        version=__version__,
        detail={"errors": STATE["errors"], **({"model": STATE["scorer"].metadata} if model_ok else {})},
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> dict[str, Any]:
    """Lightweight in-process metrics (Prometheus scraping would replace this in prod)."""
    latency = {
        path: {
            "count": len(values),
            "p50_ms": round(float(np.percentile(values, 50)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2),
        }
        for path, values in LATENCIES.items()
        if values
    }
    return {"counters": dict(COUNTERS), "latency": latency}


@app.get("/model/info", tags=["ops"])
def model_info() -> dict[str, Any]:
    return _scorer().metadata


# ------------------------------------------------------------------ scoring
@app.post("/predict_churn", response_model=PredictionResponse, tags=["scoring"])
def predict_churn(customer: CustomerPayload) -> PredictionResponse:
    """Score one customer: calibrated probability, risk band, offer economics, SHAP drivers."""
    result = _scorer().explain_one(customer.model_dump(exclude_none=False))
    return PredictionResponse(**result)


@app.post("/predict_churn/batch", response_model=BatchPredictionResponse, tags=["scoring"])
def predict_churn_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to 1,000 customers in one call."""
    import pandas as pd

    scorer = _scorer()
    frame = pd.DataFrame([c.model_dump(exclude_none=False) for c in payload.customers])
    scored = scorer.score_frame(frame)

    predictions = []
    for i, row in scored.iterrows():
        drivers = []
        if payload.include_drivers:
            drivers = scorer.explain_one(payload.customers[i].model_dump())["top_drivers"]
        predictions.append(
            PredictionResponse(
                customer_id=row["customerID"],
                churn_probability=round(float(row["churn_probability"]), 4),
                risk_band=row["risk_band"],
                decision_threshold=scorer.threshold,
                expected_value_of_offer=float(row["expected_value_of_offer"]),
                targeted_by_policy=bool(row["targeted_by_policy"]),
                recommended_action=row["recommended_action"],
                top_drivers=drivers,
                model=scorer.bundle.get("model_name"),
            )
        )
    return BatchPredictionResponse(
        count=len(predictions),
        targeted=int(scored["targeted_by_policy"].sum()),
        total_expected_value=round(float(scored.loc[scored["targeted_by_policy"] == 1,
                                                    "expected_value_of_offer"].sum()), 2),
        predictions=predictions,
    )


# -------------------------------------------------------------- intelligence
@app.post("/ask", response_model=AskResponse, tags=["intelligence"])
def ask(request: AskRequest) -> AskResponse:
    """Ask the RAG business analyst a question over structured + unstructured evidence."""
    answer = _assistant().ask(request.question, top_k=request.top_k, sources=request.sources)
    return AskResponse(**answer.to_dict())


@app.get("/kpis", tags=["analytics"])
def kpis() -> dict[str, Any]:
    cfg = load_config()
    try:
        return {
            "executive": warehouse.executive_kpis(cfg).to_dict(orient="records")[0],
            "by_contract": warehouse.segment_risk(cfg).to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Warehouse not ready: {exc}") from exc


@app.get("/customers/high-risk", tags=["analytics"])
def high_risk(threshold: float = 0.55, limit: int = 50) -> dict[str, Any]:
    cfg = load_config()
    try:
        rows = warehouse.high_risk_customers(threshold=threshold, limit=limit, cfg=cfg)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Predictions table not ready: {exc}") from exc
    return {
        "threshold": threshold,
        "count": len(rows),
        "total_margin_at_risk": round(float(rows["annual_margin_at_risk"].sum()), 2) if len(rows) else 0.0,
        "customers": rows.to_dict(orient="records"),
    }


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    cfg = load_config()
    uvicorn.run("intelliops.api.main:app", host=cfg["api.host"], port=int(cfg["api.port"]), reload=False)


if __name__ == "__main__":  # pragma: no cover
    run()
