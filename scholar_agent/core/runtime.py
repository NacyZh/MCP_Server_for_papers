"""Process-wide runtime state for cooperative shutdown."""

from __future__ import annotations

import threading

from scholar_agent.core.logging import get_logger

logger = get_logger(__name__)

_SHUTDOWN_REQUESTED = threading.Event()


def request_shutdown(reason: str = "") -> None:
    """Ask long-running worker loops to stop at their next checkpoint."""
    if not _SHUTDOWN_REQUESTED.is_set():
        logger.info("shutdown requested%s", f": {reason}" if reason else "")
    _SHUTDOWN_REQUESTED.set()


def shutdown_requested() -> bool:
    """Return whether process shutdown has been requested."""
    return _SHUTDOWN_REQUESTED.is_set()


def clear_shutdown_request() -> None:
    """Reset shutdown state for tests."""
    _SHUTDOWN_REQUESTED.clear()
