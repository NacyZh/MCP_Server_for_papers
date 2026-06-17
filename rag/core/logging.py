"""Application logging setup for the RAG MCP server."""

from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "rag" / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "rag.log"

_CONFIGURED = False
_LOG_FILE: Optional[Path] = None
_DOTENV_LOADED = False
_EXTERNAL_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "py.warnings")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b(\s*[=:]\s*)([^\s,'\"]+)"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{8,})\b"),
)


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


class _WindowsSafeRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler that does not leak rollover lock errors to stderr."""

    def doRollover(self) -> None:  # noqa: N802
        try:
            super().doRollover()
        except PermissionError:
            self.mode = "a"
            if self.stream is None:
                self.stream = self._open()

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        if isinstance(sys.exc_info()[1], PermissionError):
            return
        super().handleError(record)


class _RedactingFilter(logging.Filter):
    """Remove common secret values from application log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _redact(text: str) -> str:
    value = str(text or "")
    value = _SECRET_PATTERNS[0].sub(r"\1\2[REDACTED]", value)
    value = _SECRET_PATTERNS[1].sub("[REDACTED]", value)
    return value


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
        env_file = os.getenv("RAG_LOG_FILE")
        if env_file:
            path = Path(env_file)
        else:
            log_dir = Path(os.getenv("RAG_LOG_DIR", str(DEFAULT_LOG_DIR)))
            path = log_dir / DEFAULT_LOG_FILE.name
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def configure_logging(
    *,
    log_file: str | os.PathLike[str] | None = None,
    level: str | int | None = None,
    console: bool | None = None,
    force: bool = False,
) -> Path:
    """Configure process-wide RAG MCP logging."""

    global _CONFIGURED, _LOG_FILE
    _load_dotenv_once()
    if _CONFIGURED and not force:
        return _LOG_FILE or _resolve_log_file(log_file)

    resolved_file = _resolve_log_file(log_file)
    resolved_file.parent.mkdir(parents=True, exist_ok=True)

    raw_level = level if level is not None else os.getenv("RAG_LOG_LEVEL", "INFO")
    log_level = raw_level if isinstance(raw_level, int) else getattr(logging, str(raw_level).upper(), logging.INFO)
    use_console = _env_bool("RAG_LOG_TO_CONSOLE", False) if console is None else console
    max_bytes = _env_int("RAG_LOG_MAX_BYTES", 10 * 1024 * 1024)
    backup_count = _env_int("RAG_LOG_BACKUP_COUNT", 5)
    redacting_filter = _RedactingFilter()

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s "
        "[pid=%(process)d tid=%(threadName)s] %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("rag")
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

    file_handler = _WindowsSafeRotatingFileHandler(
        filename=str(resolved_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redacting_filter)
    root_logger.addHandler(file_handler)

    if use_console:
        console_handler = _UnicodeSafeStreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(redacting_filter)
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
    root_logger.info(
        "logging configured file=%s level=%s console=%s",
        resolved_file,
        logging.getLevelName(log_level),
        use_console,
    )
    return resolved_file


def get_logger(name: str = "rag") -> logging.Logger:
    """Return a configured RAG MCP logger."""

    configure_logging()
    if name == "rag" or name.startswith("rag."):
        return logging.getLogger(name)
    return logging.getLogger(f"rag.{name}")


def get_log_file() -> Path:
    """Return the active log file path, configuring logging if needed."""

    return configure_logging()
