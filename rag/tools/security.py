"""Input validation helpers for MCP tool boundaries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List


class ToolSecurityError(ValueError):
    """Validation error that maps cleanly to a structured tool failure."""

    def __init__(self, message: str, error_code: str, suggestion: str = ""):
        super().__init__(message)
        self.error_code = error_code
        self.suggestion = suggestion


_LOCAL_ID_RE = re.compile(r"^local_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_JOB_ID_RE = re.compile(r"^job_[A-Za-z0-9_-]{1,96}$")


def resolve_safe_pdf_filename(filename: str, base_dir: str) -> Path:
    """Resolve a user-supplied PDF filename under ``base_dir``.

    Only plain filenames are accepted. Absolute paths, nested relative paths,
    drive-qualified paths and traversal fragments are rejected before any file
    operation touches the filesystem.
    """
    raw = str(filename or "").strip()
    if not raw:
        raise ToolSecurityError(
            "filename must not be empty",
            "EMPTY_FILENAME",
            "Pass the exact PDF filename located directly in PAPERS_DIR.",
        )
    if any(ch in raw for ch in ("\x00", "\r", "\n", "/", "\\")) or ":" in raw:
        raise ToolSecurityError(
            "filename must be a plain PDF filename, not a path",
            "PATH_TRAVERSAL_BLOCKED",
            "Copy the PDF directly into PAPERS_DIR and pass only the filename.",
        )
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.name != raw or raw in {".", ".."} or ".." in candidate.parts:
        raise ToolSecurityError(
            "filename must be a plain PDF filename, not a path",
            "PATH_TRAVERSAL_BLOCKED",
            "Copy the PDF directly into PAPERS_DIR and pass only the filename.",
        )
    if candidate.suffix.lower() != ".pdf":
        raise ToolSecurityError(
            "filename must end with .pdf",
            "INVALID_FILE_TYPE",
            "Place a PDF in PAPERS_DIR and pass only its filename.",
        )

    base = Path(base_dir).expanduser().resolve()
    resolved = (base / candidate.name).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ToolSecurityError(
            "resolved PDF path escapes PAPERS_DIR",
            "PATH_TRAVERSAL_BLOCKED",
            "Copy the PDF directly into PAPERS_DIR and pass only the filename.",
        ) from exc
    return resolved


def resolve_safe_papers_subdir(subdir: str, base_dir: str) -> Path:
    """Resolve an optional papers subdirectory under ``base_dir``.

    Empty input means the base papers directory itself. Absolute paths,
    drive-qualified paths and traversal fragments are rejected before the path
    is used for directory scanning.
    """
    raw = str(subdir or "").strip()
    base = Path(base_dir).expanduser().resolve()
    if not raw:
        return base
    if any(ch in raw for ch in ("\x00", "\r", "\n")) or ":" in raw:
        raise ToolSecurityError(
            "subdir must be a clean relative directory under PAPERS_DIR",
            "PATH_TRAVERSAL_BLOCKED",
            "Use an empty subdir for PAPERS_DIR itself, or a relative child directory name.",
        )
    candidate = Path(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ToolSecurityError(
            "subdir must be a relative directory under PAPERS_DIR",
            "PATH_TRAVERSAL_BLOCKED",
            "Use an empty subdir for PAPERS_DIR itself, or a relative child directory name.",
        )
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ToolSecurityError(
            "resolved directory escapes PAPERS_DIR",
            "PATH_TRAVERSAL_BLOCKED",
            "Use a directory located inside PAPERS_DIR.",
        ) from exc
    return resolved


def validate_text_length(value: str, *, field: str, max_chars: int, error_code: str) -> str:
    text = str(value or "").strip()
    if len(text) > int(max_chars):
        raise ToolSecurityError(
            f"{field} is too long: {len(text)} characters, max {max_chars}",
            error_code,
            f"Shorten {field} to at most {max_chars} characters.",
        )
    return text


def parse_local_ids(raw_ids: str | Iterable[str], *, max_ids: int) -> List[str]:
    if isinstance(raw_ids, str):
        ids = [pid.strip() for pid in raw_ids.split(",") if pid.strip()]
    else:
        ids = [str(pid or "").strip() for pid in raw_ids if str(pid or "").strip()]
    if len(ids) > int(max_ids):
        raise ToolSecurityError(
            f"Too many paper IDs: {len(ids)}, max {max_ids}",
            "TOO_MANY_PAPER_IDS",
            f"Pass at most {max_ids} local paper IDs in one call.",
        )
    invalid = [pid for pid in ids if not _LOCAL_ID_RE.match(pid)]
    if invalid:
        raise ToolSecurityError(
            f"Invalid local paper ID: {invalid[0]}",
            "INVALID_LOCAL_ID",
            "Use IDs returned by list_local_database or search_local_database.",
        )
    return ids


def validate_local_id(local_id: str) -> str:
    value = str(local_id or "").strip()
    if not value:
        raise ToolSecurityError(
            "local_id must not be empty",
            "EMPTY_LOCAL_ID",
            "Use list_local_database or search_local_database to find a valid local_id.",
        )
    if not _LOCAL_ID_RE.match(value):
        raise ToolSecurityError(
            f"Invalid local paper ID: {value}",
            "INVALID_LOCAL_ID",
            "Use IDs returned by list_local_database or search_local_database.",
        )
    return value


def validate_job_id(job_id: str) -> str:
    value = str(job_id or "").strip()
    if not value:
        raise ToolSecurityError(
            "job_id must not be empty",
            "EMPTY_JOB_ID",
            "Use the exact job_id returned by a background tool.",
        )
    if not _JOB_ID_RE.match(value):
        raise ToolSecurityError(
            f"Invalid job ID: {value}",
            "INVALID_JOB_ID",
            "Use the exact job_id returned by a background tool.",
        )
    return value


def resolve_project_file(path_value: str, project_root: str, *, suffix: str = "") -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise ToolSecurityError(
            "path must not be empty",
            "EMPTY_PATH",
            "Provide a project-relative path to an existing evaluation dataset.",
        )
    if any(ch in raw for ch in ("\x00", "\r", "\n")):
        raise ToolSecurityError(
            "path contains invalid control characters",
            "INVALID_PATH",
            "Provide a clean project-relative dataset path.",
        )
    root = Path(project_root).expanduser().resolve()
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ToolSecurityError(
            "path escapes the project root",
            "PATH_TRAVERSAL_BLOCKED",
            "Store evaluation datasets inside the project and pass a project-relative path.",
        ) from exc
    if suffix and resolved.suffix.lower() != suffix.lower():
        raise ToolSecurityError(
            f"path must end with {suffix}",
            "INVALID_FILE_TYPE",
            f"Use a {suffix} evaluation dataset.",
        )
    return resolved


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    value = str(text or "")
    limit = int(max_chars)
    if len(value) <= limit:
        return value, False
    suffix = "\n\n[Result truncated by TOOL_MAX_RETURN_CHARS.]"
    return value[: max(0, limit - len(suffix))].rstrip() + suffix, True
