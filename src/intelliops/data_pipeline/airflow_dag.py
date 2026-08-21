"""Airflow DAG wrapper for the IntelliOps platform.

Drop this file (or a symlink to it) into ``$AIRFLOW_HOME/dags/``. It imports the
same functions the CLI calls — the orchestrator adds scheduling, retries and
alerting, it does not re-implement the pipeline. That separation is the reason
the platform is testable without Airflow installed at all.

Schedule: daily at 02:00 UTC. Scoring runs after training so the prediction table
is never newer than the model that produced it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:  # Airflow is an orchestration concern, not a runtime dependency of the package
    from airflow.decorators import dag, task
except ImportError:  # pragma: no cover
    dag = task = None  # type: ignore[assignment]


DEFAULT_ARGS = {
    "owner": "data-science",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "depends_on_past": False,
}


if dag is not None:  # pragma: no cover - exercised only inside Airflow

    @dag(
        dag_id="intelliops_daily",
        description="ETL → churn model → segmentation → NLP → RAG index",
        schedule="0 2 * * *",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args=DEFAULT_ARGS,
        tags=["intelliops", "ml", "customer-analytics"],
        max_active_runs=1,
    )
    def intelliops_daily():
        @task
        def etl() -> dict:
            from intelliops.data_pipeline.run_pipeline import run

            return run().summary()

        @task
        def train(_upstream: dict) -> dict:
            from intelliops.churn_model.train import train as train_model

            report = train_model()
            # Fail the run rather than publish a regressed model.
            auc = report["final_metrics"]["roc_auc"]
            if auc < 0.78:
                raise ValueError(f"Model quality gate failed: ROC-AUC {auc:.4f} < 0.78")
            return {"model": report["selected_model"], "roc_auc": auc}

        @task
        def segment(_upstream: dict) -> int:
            from intelliops.churn_model.segmentation import segment as run_segmentation

            return len(run_segmentation())

        @task
        def nlp(_upstream: dict) -> int:
            from intelliops.nlp_engine.run_nlp import run as run_nlp

            return len(run_nlp()["topics"])

        @task
        def refresh_rag_index(_a: int, _b: int) -> int:
            from intelliops.rag_assistant.retriever import build_index

            return len(build_index().documents)

        etl_result = etl()
        trained = train(etl_result)
        refresh_rag_index(segment(trained), nlp(etl_result))

    intelliops_daily()
