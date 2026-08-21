"""Self-bootstrapping: bring the platform up from an empty checkout.

A deployed demo has no `make all` step. Streamlit Community Cloud clones the repo,
installs requirements and runs the app — into a container with no warehouse, no
model and no vector index. Rather than showing a stack trace on first visit, the app
calls ``ensure_platform_ready()``, which builds whatever is missing and skips whatever
is already there.

The same function makes a local first run forgiving: someone who clones the repo and
goes straight to ``make dashboard`` gets a working page instead of an error telling
them to have run something else first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, load_config
from .data_pipeline import warehouse
from .logging_utils import get_logger, stage

logger = get_logger(__name__)


@dataclass
class BootstrapResult:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def did_work(self) -> bool:
        return bool(self.ran)

    def summary(self) -> str:
        if not self.ran:
            return "platform already built — nothing to do"
        return f"built: {', '.join(self.ran)}"


def platform_state(cfg: Config | None = None) -> dict[str, bool]:
    """What already exists. Cheap enough to call on every page load."""
    cfg = cfg or load_config()
    tables = cfg["warehouse.schema_tables"]
    return {
        "warehouse": warehouse.table_exists(tables["features"], cfg),
        "model": cfg.path("api.model_path").exists(),
        "predictions": warehouse.table_exists(tables["predictions"], cfg),
        "segments": warehouse.table_exists("fct_customer_segments", cfg),
        "nlp": warehouse.table_exists("agg_topic_summary", cfg),
        "rag_index": cfg.path("rag.vector_store").exists(),
    }


def ensure_platform_ready(cfg: Config | None = None, include_rag: bool = True,
                          force: bool = False) -> BootstrapResult:
    """Build any missing stage, in dependency order. Idempotent and safe to re-call."""
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    state = platform_state(cfg)
    result = BootstrapResult()

    def step(name: str, already_done: bool, fn) -> None:
        if already_done and not force:
            result.skipped.append(name)
            return
        with stage(logger, f"bootstrap: {name}"):
            fn()
        result.ran.append(name)

    from .churn_model.segmentation import segment
    from .churn_model.train import train
    from .data_pipeline.run_pipeline import run as run_pipeline
    from .nlp_engine.run_nlp import run as run_nlp

    step("warehouse", state["warehouse"], lambda: run_pipeline(cfg))
    # Training also writes the prediction table, so both gates guard the same step.
    step("model", state["model"] and state["predictions"], lambda: train(cfg))
    step("segments", state["segments"], lambda: segment(cfg))
    step("nlp", state["nlp"], lambda: run_nlp(cfg))

    if include_rag:
        from .rag_assistant.retriever import build_index

        step("rag_index", state["rag_index"], lambda: build_index(cfg))

    logger.info("Bootstrap complete — %s", result.summary())
    return result
