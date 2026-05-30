"""Application logging setup.

This module is the single runtime logging entry point for ScholarAgent.
It configures a standard-library logger with a rotating file handler and
an optional console handler.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "workspace" / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "scholar_agent.log"

_CONFIGURED = False
_LOG_FILE: Optional[Path] = None
_DOTENV_LOADED = False
_EXTERNAL_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "py.warnings")


class _UnicodeSafeStreamHandler(logging.StreamHandler):
    """Stream handler that degrades gracefully on narrow console encodings."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                message = self.format(record)
                stream = self.stream
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_dotenv_once() -> None:
    """Load project .env once so log env vars work before config imports."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass
    _DOTENV_LOADED = True


def _resolve_log_file(log_file: str | os.PathLike[str] | None = None) -> Path:
    if log_file:
        path = Path(log_file)
    else:
        env_file = os.getenv("SCHOLAR_AGENT_LOG_FILE")
        if env_file:
            path = Path(env_file)
        else:
            log_dir = Path(os.getenv("SCHOLAR_AGENT_LOG_DIR", str(DEFAULT_LOG_DIR)))
            path = log_dir / DEFAULT_LOG_FILE.name
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def configure_logging(
    *,
    log_file: str | os.PathLike[str] | None = None,
    level: str | int | None = None,
    console: bool | None = None,
    force: bool = False,
) -> Path:
    """Configure process-wide ScholarAgent logging.

    Environment variables:
    - ``SCHOLAR_AGENT_LOG_FILE``: full log file path.
    - ``SCHOLAR_AGENT_LOG_DIR``: directory used when no explicit file is set.
    - ``SCHOLAR_AGENT_LOG_LEVEL``: DEBUG, INFO, WARNING, ERROR.
    - ``SCHOLAR_AGENT_LOG_TO_CONSOLE``: true/false.
    - ``SCHOLAR_AGENT_LOG_MAX_BYTES``: rotating file size.
    - ``SCHOLAR_AGENT_LOG_BACKUP_COUNT``: number of rotated files to keep.
    """
    global _CONFIGURED, _LOG_FILE
    _load_dotenv_once()
    if _CONFIGURED and not force:
        return _LOG_FILE or _resolve_log_file(log_file)

    resolved_file = _resolve_log_file(log_file)
    resolved_file.parent.mkdir(parents=True, exist_ok=True)

    raw_level = level if level is not None else os.getenv("SCHOLAR_AGENT_LOG_LEVEL", "INFO")
    log_level = raw_level if isinstance(raw_level, int) else getattr(logging, str(raw_level).upper(), logging.INFO)
    use_console = _env_bool("SCHOLAR_AGENT_LOG_TO_CONSOLE", True) if console is None else console
    max_bytes = _env_int("SCHOLAR_AGENT_LOG_MAX_BYTES", 10 * 1024 * 1024)
    backup_count = _env_int("SCHOLAR_AGENT_LOG_BACKUP_COUNT", 5)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s "
        "[pid=%(process)d tid=%(threadName)s] %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("scholar_agent")
    root_logger.setLevel(log_level)
    root_logger.propagate = False

    handlers_to_close = set(root_logger.handlers)
    for logger_name in _EXTERNAL_LOGGERS:
        handlers_to_close.update(logging.getLogger(logger_name).handlers)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for logger_name in _EXTERNAL_LOGGERS:
        external_logger = logging.getLogger(logger_name)
        for handler in list(external_logger.handlers):
            external_logger.removeHandler(handler)
    for handler in handlers_to_close:
        handler.close()

    file_handler = RotatingFileHandler(
        filename=str(resolved_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if use_console:
        console_handler = _UnicodeSafeStreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    for logger_name in _EXTERNAL_LOGGERS:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(log_level)
        external_logger.propagate = False
        for handler in root_logger.handlers:
            external_logger.addHandler(handler)

    logging.captureWarnings(True)
    _CONFIGURED = True
    _LOG_FILE = resolved_file
    root_logger.info("logging configured file=%s level=%s console=%s", resolved_file, logging.getLevelName(log_level), use_console)
    return resolved_file


def get_logger(name: str = "scholar_agent") -> logging.Logger:
    """Return a configured ScholarAgent logger."""
    configure_logging()
    if name == "scholar_agent" or name.startswith("scholar_agent."):
        return logging.getLogger(name)
    return logging.getLogger(f"scholar_agent.{name}")


def get_log_file() -> Path:
    """Return the active log file path, configuring logging if needed."""
    return configure_logging()
