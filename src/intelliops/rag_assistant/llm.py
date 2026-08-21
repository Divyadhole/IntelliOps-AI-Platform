"""LLM provider abstraction.

Resolution: Anthropic → OpenAI → none. When no key or SDK is present, ``generate``
returns ``None`` and the assistant falls back to deterministic synthesis, so the
demo still answers questions on a laptop with no API credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import Config, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClient:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.provider, self.model, self._client = self._resolve()

    # ------------------------------------------------------------- resolution
    def _resolve(self) -> tuple[str, str, object | None]:
        preference = self.cfg.get("rag.llm.provider", "auto")

        if preference in {"auto", "anthropic"} and os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                model = self.cfg.get("rag.llm.anthropic_model", "claude-sonnet-4-5")
                logger.info("LLM provider: anthropic (%s)", model)
                return "anthropic", model, anthropic.Anthropic()
            except ImportError:
                logger.warning("ANTHROPIC_API_KEY set but the `anthropic` package is not installed")

        if preference in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                model = self.cfg.get("rag.llm.openai_model", "gpt-4o-mini")
                logger.info("LLM provider: openai (%s)", model)
                return "openai", model, OpenAI()
            except ImportError:
                logger.warning("OPENAI_API_KEY set but the `openai` package is not installed")

        logger.warning("No LLM provider configured — the assistant will use deterministic synthesis. "
                       "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for generative answers.")
        return "none", "", None

    @property
    def available(self) -> bool:
        return self._client is not None

    # -------------------------------------------------------------- generate
    def generate(self, system: str, user: str) -> LLMResponse | None:
        if not self.available:
            return None
        max_tokens = int(self.cfg.get("rag.llm.max_tokens", 900))
        temperature = float(self.cfg.get("rag.llm.temperature", 0.2))
        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model, max_tokens=max_tokens, temperature=temperature,
                    system=system, messages=[{"role": "user", "content": user}],
                )
                return LLMResponse("".join(b.text for b in resp.content if b.type == "text"),
                                   self.provider, self.model)
            resp = self._client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return LLMResponse(resp.choices[0].message.content or "", self.provider, self.model)
        except Exception as exc:
            logger.error("LLM call failed (%s: %s) — falling back to deterministic synthesis",
                         exc.__class__.__name__, exc)
            return None
