"""Observability helpers for tool and job execution."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from rag.core.logging import get_logger

logger = get_logger(__name__)


def new_request_id(prefix: str = "req") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextmanager
def timed_span(operation: str, **fields) -> Iterator[dict]:
    """Log start/end events with elapsed milliseconds."""

    start = time.perf_counter()
    context = {"operation": operation, **fields}
    logger.info("[obs] start %s", _format_fields(context))
    try:
        yield context
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("[obs] error %s", _format_fields({**context, "elapsed_ms": elapsed_ms}))
        raise
    else:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("[obs] end %s", _format_fields({**context, "elapsed_ms": elapsed_ms}))


def log_event(event: str, **fields) -> None:
    logger.info("[obs] %s %s", event, _format_fields(fields))


def _format_fields(fields: dict) -> str:
    return " ".join(f"{key}={value!r}" for key, value in fields.items() if value not in (None, ""))
