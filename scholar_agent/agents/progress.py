"""Lightweight progress event bridge for streaming agent runs."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

ProgressSink = Callable[[str, dict[str, Any]], None]

_progress_sink: ContextVar[ProgressSink | None] = ContextVar("scholar_agent_progress_sink", default=None)


def set_progress_sink(sink: ProgressSink | None):
    """Set the progress sink for the current execution context."""
    return _progress_sink.set(sink)


def reset_progress_sink(token) -> None:
    """Restore the previous progress sink."""
    _progress_sink.reset(token)


def emit_progress(event: str, **data: Any) -> None:
    """Emit a progress event if the current run is streaming."""
    sink = _progress_sink.get()
    if sink is None:
        return
    try:
        sink(event, data)
    except Exception:
        # Progress reporting must never break the agent workflow.
        return
