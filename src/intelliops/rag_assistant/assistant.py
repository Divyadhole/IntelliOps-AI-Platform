"""The RAG business analyst: question in, grounded answer with citations out.

Guardrails that make this different from a chatbot bolted onto a vector DB:

* the system prompt forbids any number that is not in the retrieved evidence;
* every claim carries an ``[E1]``-style citation resolvable to a warehouse row or
  a verbatim customer message;
* if no LLM is configured, a deterministic synthesiser produces the same shape of
  answer from the structured evidence, so the endpoint never returns an apology.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from ..logging_utils import get_logger
from .llm import LLMClient
from .retriever import load_index
from .vector_store import Document, VectorStore

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior customer-analytics partner at a subscription business.
You answer executives' questions using ONLY the numbered evidence provided.

Rules:
- Every quantitative claim must cite its evidence id, e.g. [E3]. Never state a number that is not in the evidence.
- If the evidence cannot answer the question, say exactly what is missing and what data would settle it.
- Distinguish correlation from causation. Feedback themes are associated with churn; they are not proven causes.
- Be concise and decision-oriented.

Structure your answer as:
**Answer** — two or three sentences.
**What the data shows** — 3-5 bullets, each cited.
**Recommended actions** — 2-3 concrete actions, each tied to the driver it addresses.
**Confidence & caveats** — one or two lines on how far the evidence stretches.
"""


@dataclass
class AssistantAnswer:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    generated_by: str = "deterministic"
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "generated_by": self.generated_by,
            "model": self.model,
        }


# Executives and analysts use different vocabulary from the corpus. Without
# expansion, "why are customers leaving?" scores ~0 against a corpus that says
# "churn", "cancel" and "switched" — a classic vocabulary-mismatch retrieval failure.
QUERY_EXPANSIONS = {
    "leaving": "churn cancel cancelled attrition switch left",
    "leave": "churn cancel attrition switch",
    "churn": "leaving cancel attrition retention",
    "unhappy": "complaint negative dissatisfied poor bad",
    "complaints": "negative unhappy issue problem",
    "revenue": "charges billing monthly arpu margin",
    "price": "charges cost billing expensive increase",
    "support": "ticket agent help service technical",
    "delivery": "shipment late delayed warehouse",
    "risk": "churn probability high-risk retention",
    "loyal": "tenure long-standing contract retention",
}


def expand_query(question: str) -> str:
    """Append domain synonyms for any expansion key present in the question."""
    lowered = question.lower()
    extras = [syn for key, syn in QUERY_EXPANSIONS.items() if key in lowered]
    return f"{question} {' '.join(extras)}".strip() if extras else question


def _format_evidence(hits: list[tuple[Document, float]]) -> tuple[str, list[dict[str, Any]]]:
    lines, citations = [], []
    for i, (doc, score) in enumerate(hits, start=1):
        tag = f"E{i}"
        lines.append(f"[{tag}] ({doc.source}) {doc.text}")
        citations.append(
            {"id": tag, "doc_id": doc.doc_id, "source": doc.source,
             "score": round(score, 4), "text": doc.text[:400]}
        )
    return "\n".join(lines), citations


def _deterministic_answer(question: str, hits: list[tuple[Document, float]]) -> str:
    """Structured synthesis from retrieved evidence — used when no LLM is configured."""
    by_source: dict[str, list[Document]] = {}
    for doc, _ in hits:
        by_source.setdefault(doc.source, []).append(doc)

    parts = [f"**Answer** — Based on {len(hits)} retrieved evidence items for: “{question}”."]

    if "kpi" in by_source:
        parts.append("\n**Portfolio position**")
        for doc in by_source["kpi"][:3]:
            parts.append(f"- {doc.text} [E{[d for d, _ in hits].index(doc) + 1}]")

    if "model_driver" in by_source:
        parts.append("\n**What the churn model attributes risk to**")
        for doc in by_source["model_driver"][:4]:
            parts.append(f"- {doc.text} [E{[d for d, _ in hits].index(doc) + 1}]")

    if "topic_summary" in by_source:
        # Without a language model to weigh relevance, rank themes by how negative
        # they are — a "why are customers unhappy" answer should not lead with praise.
        themes = sorted(by_source["topic_summary"],
                        key=lambda d: -float(d.metadata.get("negative_share", 0.0)))
        parts.append("\n**What customers are talking about (most negative themes first)**")
        for doc in themes[:4]:
            parts.append(f"- {doc.text} [E{[d for d, _ in hits].index(doc) + 1}]")

    if "review" in by_source:
        parts.append("\n**Representative verbatims**")
        for doc in by_source["review"][:3]:
            parts.append(f"- “{doc.text[:220]}” [E{[d for d, _ in hits].index(doc) + 1}]")

    parts.append(
        "\n**Confidence & caveats** — This answer was assembled deterministically from retrieved "
        "records because no LLM provider is configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "for a synthesised narrative. Associations shown are not causal evidence."
    )
    return "\n".join(parts)


class BusinessAnalystAssistant:
    def __init__(self, cfg: Config | None = None, store: VectorStore | None = None,
                 llm: LLMClient | None = None) -> None:
        self.cfg = cfg or load_config()
        self.store = store or load_index(self.cfg)
        self.llm = llm or LLMClient(self.cfg)

    # Evidence budget per source. Structured facts anchor the numbers; verbatims
    # supply the human detail. Proportions matter more than the total.
    DEFAULT_QUOTAS = {"kpi": 3, "model_driver": 3, "topic_summary": 6, "review": 4}

    def ask(self, question: str, top_k: int | None = None,
            sources: list[str] | None = None) -> AssistantAnswer:
        retrieval_query = expand_query(question)
        if sources:
            top_k = top_k or int(self.cfg.get("rag.top_k", 6))
            hits = self.store.search(retrieval_query, top_k=top_k, sources=sources)
        else:
            quotas = dict(self.DEFAULT_QUOTAS)
            if top_k:  # scale the budget while keeping the mix
                scale = top_k / sum(quotas.values())
                quotas = {k: max(1, round(v * scale)) for k, v in quotas.items()}
            hits = self.store.search_stratified(retrieval_query, quotas)
        if not hits:
            return AssistantAnswer(question, "No relevant evidence was retrieved for that question.")

        evidence, citations = _format_evidence(hits)
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence}"

        response = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        if response is None:
            return AssistantAnswer(question, _deterministic_answer(question, hits), citations)
        return AssistantAnswer(question, response.text.strip(), citations, "llm", response.model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the IntelliOps business analyst a question")
    parser.add_argument("question", nargs="+", help="e.g. 'why are customers leaving?'")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    args = parser.parse_args()

    answer = BusinessAnalystAssistant().ask(" ".join(args.question), top_k=args.top_k)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
    else:
        print("\n" + answer.answer + "\n")
        print("Sources:")
        for c in answer.citations:
            print(f"  [{c['id']}] {c['source']}:{c['doc_id']} (score {c['score']})")


if __name__ == "__main__":
    main()
