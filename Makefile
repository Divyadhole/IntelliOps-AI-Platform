.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python3            # override with `make PY=python` inside a venv on Windows
export PYTHONPATH := src

.PHONY: help install install-dev install-nlp pipeline train segment nlp rag-index all api dashboard report mlflow ask test lint format docker-build docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:  ## Install runtime + dev/test dependencies
	$(PY) -m pip install -r requirements-dev.txt

install-nlp:  ## Install the optional transformer / LLM stack
	$(PY) -m pip install -r requirements-nlp.txt

pipeline:  ## Module 1 — run the ETL pipeline into the warehouse
	$(PY) -m intelliops.data_pipeline.run_pipeline

train:  ## Module 2 — train, calibrate, explain and score the churn model
	$(PY) -m intelliops.churn_model.train

segment:  ## Module 2b — customer segmentation
	$(PY) -m intelliops.churn_model.segmentation

nlp:  ## Module 3a — sentiment + topic discovery over customer feedback
	$(PY) -m intelliops.nlp_engine.run_nlp

rag-index:  ## Module 3b — build the RAG vector index
	$(PY) -m intelliops.rag_assistant.retriever

all: pipeline train segment nlp rag-index report  ## Run the full platform end to end
	@echo "✔ Platform ready — run 'make api' and 'make dashboard'"

api:  ## Module 4 — serve the FastAPI app (docs at /docs)
	$(PY) -m uvicorn intelliops.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:  ## Module 4 — launch the interactive Streamlit dashboard
	streamlit run dashboard/app.py

report:  ## Module 4 — build the shareable static dashboard (artifacts/reports/dashboard.html)
	$(PY) -m intelliops.reporting.build_dashboard

mlflow:  ## Open the MLflow experiment tracking UI
	mlflow ui --backend-store-uri ./artifacts/mlruns --port 5000

ask:  ## Ask the RAG analyst, e.g. make ask Q="why are customers leaving?"
	$(PY) -m intelliops.rag_assistant.assistant "$(Q)"

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests dashboard

format:  ## Auto-format with black + ruff --fix
	$(PY) -m black src tests dashboard
	$(PY) -m ruff check --fix src tests dashboard

docker-build:  ## Build the API and dashboard images
	docker compose build

docker-up:  ## Start Postgres, MLflow, the API and the dashboard
	docker compose up -d

docker-down:  ## Stop the stack
	docker compose down

clean:  ## Remove generated data, models and caches
	rm -rf data/processed/* data/interim/* artifacts/models/* artifacts/reports/* artifacts/figures/* artifacts/mlruns
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✔ Cleaned (re-run 'make all' to rebuild)"
