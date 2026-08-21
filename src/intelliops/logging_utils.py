"""Structured logging used by every pipeline stage.

Pipelines are judged on whether a failure is diagnosable at 3am, so every stage
logs a start line, a row-count line, and a completion line with elapsed time.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)


@contextmanager
def stage(logger: logging.Logger, name: str) -> Iterator[None]:
    """Log the boundaries and duration of a pipeline stage."""
    logger.info("▶ %s — start", name)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("✖ %s — FAILED after %.2fs", name, time.perf_counter() - started)
        raise
    logger.info("✔ %s — done in %.2fs", name, time.perf_counter() - started)
