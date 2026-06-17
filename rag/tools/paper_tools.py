"""Concrete tool implementations for ScholarAgent.

Tools exposed:
- EvidenceChunkRetrievalTool: Retrieve supporting evidence chunks
- PaperOutlineTool: Return section outline metadata
- PaperProfileTool: Return cached compact paper profile
- PaperSummaryTool: Return cached structured paper summary
- BuildPaperSummaryTool: Build summary/profile cache as a background job
- DbListTool: List all papers in local database
- DbSearchTool: Search paper metadata records in local database
- DbAddTool: Parse and ingest a PDF (with duplicate policy)
- DbImportDirectoryTool: Batch import PDFs from PAPERS_DIR
- ToolJobStatusTool: Check status/result for long-running jobs
- DbDeleteTool: Delete from SQLite + ChromaDB
- DedupDatabaseTool: Scan the database for duplicate papers
- BackfillMetadataTool: Recompute dedup columns for legacy rows
- RetrievalQualityEvalTool: Run retrieval quality evaluation as a job
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from config import conf
from rag.core.logging import get_logger
from rag.storage import PaperManager
from rag.tools.base import BaseTool, ToolResult
from rag.tools.jobs import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    JobManager,
    job_manager,
)
from rag.tools.security import (
    ToolSecurityError,
    parse_local_ids,
    resolve_project_file,
    resolve_safe_papers_subdir,
    resolve_safe_pdf_filename,
    truncate_text,
    validate_job_id,
    validate_local_id,
    validate_text_length,
)

logger = get_logger(__name__)

summary_job_manager = JobManager(
    max_workers=conf.RAG_SUMMARY_CONCURRENCY,
    name="summary",
    thread_name_prefix="rag-summary-job",
)

if TYPE_CHECKING:
    from rag.storage import PaperDB


def _classify_ingest_failure(message: str) -> str:
    text = str(message or "").lower()
    if "stage=hash_pdf" in text:
        return "PDF_READ_FAILED"
    if "stage=parse_pdf" in text:
        return "PDF_PARSE_FAILED"
    if "stage=vector_store" in text:
        return "VECTOR_INDEX_FAILED"
    if "stage=sqlite_store" in text:
        return "SQLITE_STORE_FAILED"
    if "duplicate" in text:
        return "DUPLICATE_PAPER"
    if "file not found" in text:
        return "PDF_NOT_FOUND"
    if "torch" in text or "cuda" in text:
        return "MODEL_RUNTIME_ERROR"
    if "model" in text or "huggingface" in text:
        return "MODEL_UNAVAILABLE"
    if "extract text" in text or "parser" in text:
        return "PDF_PARSE_FAILED"
    return "PDF_INGESTION_FAILED"


def _suggest_for_ingest_failure(message: str) -> str:
    code = _classify_ingest_failure(message)
    if code == "DUPLICATE_PAPER":
        return "Use on_duplicate='replace' to replace the existing paper, or 'keep_both' to import another copy."
    if code == "MODEL_RUNTIME_ERROR":
        return "Check PAPER_PARSER_DEVICE and whether the active PyTorch build supports CUDA."
    if code == "MODEL_UNAVAILABLE":
        return "Check BGE model paths or enable model auto-download."
    if code == "PDF_PARSE_FAILED":
        return (
            "Check whether the PDF is text-based, whether marker dependencies are installed, "
            "and whether PAPER_PARSER_DEVICE matches the active PyTorch runtime."
        )
    if code == "PDF_READ_FAILED":
        return "Check whether the PDF exists, is readable, and is not locked by another process."
    if code == "VECTOR_INDEX_FAILED":
        return "Check BGE embedding model availability, ChromaDB health, and model runtime logs."
    if code == "SQLITE_STORE_FAILED":
        return "Check SQLite database path, permissions, schema migration, and disk health."
    if code == "PDF_NOT_FOUND":
        return "Copy the PDF into PAPERS_DIR and retry with the exact filename."
    return "Check server logs for details and retry after fixing the underlying issue."


def _coerce_tool_bool(value: Any, *, field: str) -> tuple[bool, ToolResult | None]:
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True, None
        if lowered in {"false", "0", "no", "n", ""}:
            return False, None
    if isinstance(value, int) and value in {0, 1}:
        return bool(value), None
    return False, ToolResult.fail(
        f"Invalid boolean value for {field}: {value!r}",
        error_code="INVALID_BOOLEAN",
        suggestion=f"Use true or false for {field}.",
    )


def _security_failure(exc: ToolSecurityError, data=None) -> ToolResult:
    return ToolResult.fail(
        str(exc),
        error_code=exc.error_code,
        suggestion=exc.suggestion,
        data=data,
    )


def _validate_detail_level(value: str) -> str:
    detail_level = str(value or "medium").strip().lower()
    if detail_level not in {"short", "medium", "long"}:
        raise ToolSecurityError(
            f"Invalid detail_level: {value!r}",
            "INVALID_ARGUMENT",
            "Use detail_level='short', 'medium', or 'long'.",
        )
    return detail_level


def _validate_language(value: str, *, allow_both: bool = False) -> str:
    language = str(value or "en").strip().lower()
    allowed = {"zh", "en", "both"} if allow_both else {"zh", "en"}
    if language not in allowed:
        raise ToolSecurityError(
            f"Invalid language: {value!r}",
            "INVALID_ARGUMENT",
            f"Use language from: {', '.join(sorted(allowed))}.",
        )
    return language


def _parse_detail_levels(value: Any) -> list[str]:
    if value is None or value == "":
        raw = ["short", "medium", "long"]
    elif isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw = [str(item or "").strip() for item in value if str(item or "").strip()]
    levels = [_validate_detail_level(item) for item in raw]
    return list(dict.fromkeys(levels)) or ["short", "medium", "long"]


def _summary_not_ready(paper_id: str) -> ToolResult:
    return ToolResult.fail(
        "Paper summary has not been generated yet.",
        error_code="SUMMARY_NOT_READY",
        suggestion="Call build_paper_summary for this paper_id, then poll get_tool_job_status.",
        data={
            "status": "not_ready",
            "error_code": "SUMMARY_NOT_READY",
            "suggested_tool": "build_paper_summary",
            "paper_id": paper_id,
        },
    )


def _get_tool_job_record(job_id: str) -> dict | None:
    return job_manager.get(job_id) or summary_job_manager.get(job_id)


def _observe_tool_job_record(job_id: str) -> dict | None:
    return job_manager.observe_status(job_id) or summary_job_manager.observe_status(job_id)


def _combined_job_stats() -> dict:
    default_stats = job_manager.stats()
    summary_stats = summary_job_manager.stats()
    keys = ("total", "queued", "running", "succeeded", "failed")
    return {
        **{key: default_stats[key] + summary_stats[key] for key in keys},
        "queues": {
            "default": default_stats,
            "summary": summary_stats,
        },
    }


def _job_status_wait_seconds(status_checks: int) -> float:
    initial = max(0.0, float(conf.TOOL_JOB_STATUS_WAIT_INITIAL_SECONDS))
    step = max(0.0, float(conf.TOOL_JOB_STATUS_WAIT_STEP_SECONDS))
    maximum = max(0.0, float(conf.TOOL_JOB_STATUS_WAIT_MAX_SECONDS))
    if maximum <= 0.0:
        return 0.0
    value = initial + max(0, int(status_checks) - 1) * step
    return min(value, maximum)


def _wait_for_nonterminal_job(job_id: str, record: dict) -> tuple[dict, float, float]:
    status = record.get("status")
    if status not in {JOB_STATUS_QUEUED, JOB_STATUS_RUNNING}:
        return record, 0.0, 0.0

    wait_seconds = _job_status_wait_seconds(int(record.get("status_checks") or 1))
    if wait_seconds <= 0.0:
        return record, 0.0, _job_status_wait_seconds(int(record.get("status_checks") or 1) + 1)

    deadline = time.monotonic() + wait_seconds
    current = record
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        time.sleep(min(0.25, max(0.01, remaining)))
        refreshed = _get_tool_job_record(job_id)
        if refreshed is None:
            return current, wait_seconds, 0.0
        current = refreshed
        if current.get("status") in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED}:
            break
    next_wait = 0.0
    if current.get("status") in {JOB_STATUS_QUEUED, JOB_STATUS_RUNNING}:
        next_wait = _job_status_wait_seconds(int(record.get("status_checks") or 1) + 1)
    return current, wait_seconds, next_wait


def _format_evidence_chunks(results: list[dict], max_chars: int) -> tuple[str, dict]:
    chunks = []
    remaining = max(1, int(max_chars))
    truncated = False
    for idx, res in enumerate(results):
        raw_text = str(res.get("content") or res.get("text") or "")
        text, did_truncate = truncate_text(raw_text, remaining)
        truncated = truncated or did_truncate or len(raw_text) > len(text)
        chunk = {
            "paper_id": res.get("paper_id"),
            "title": res.get("title"),
            "section": res.get("section_title") or res.get("section_name") or "Unknown",
            "section_name": res.get("section_name") or "unknown",
            "chunk_id": res.get("chunk_id") or f"{res.get('paper_id', 'unknown')}:{idx}",
            "score": res.get("score", 0.0),
            "text": text,
            "page_start": res.get("page_start"),
            "page_end": res.get("page_end"),
        }
        remaining -= len(text)
        if remaining <= 0:
            truncated = idx < len(results) - 1 or truncated
            chunks.append(chunk)
            break
        chunks.append(chunk)

    lines = ["=== Evidence Chunks ==="]
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Chunk {idx}] Paper ID: {chunk['paper_id']} | Title: {chunk['title']} | "
            f"Section: {chunk['section']} | Score: {chunk['score']}"
        )
        lines.append(str(chunk["text"]))
        lines.append("")
    data = {
        "chunks": chunks,
        "truncated": truncated,
        "next_cursor": None,
        "max_chars": max_chars,
    }
    return "\n".join(lines).strip(), data


class EvidenceChunkRetrievalTool(BaseTool):
    name = "retrieve_evidence_chunks"
    description = (
        "Retrieve exact supporting passages, citations, and evidence chunks from local papers. "
        "This tool is for verification and citation lookup; prefer get_paper_profile or "
        "get_paper_summary for general paper summarization."
    )
    params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Concise academic evidence query. Leave empty when paper_ids are provided.",
                "default": "",
            },
            "paper_ids": {
                "type": "string",
                "description": "Optional comma-separated local paper IDs, e.g. 'local_abc123,local_def456'.",
                "default": "",
            },
            "section": {
                "type": "string",
                "description": "Optional section filter, e.g. introduction, system model, method, experiments.",
                "default": "",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to return per call or per paper (1-20, default 5).",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum total characters of evidence text to return (1000-50000, default 6000).",
                "minimum": 1000,
                "maximum": 50000,
                "default": 6000,
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager()
        return self.pm

    def execute(
        self,
        query: str = "",
        paper_ids: str = "",
        section: str = "",
        top_k: int = 5,
        max_chars: int = 6000,
    ) -> ToolResult:
        try:
            top_k = max(1, min(int(top_k), 20))
        except (TypeError, ValueError):
            top_k = 5
        try:
            max_chars = max(1000, min(int(max_chars), 50000))
        except (TypeError, ValueError):
            max_chars = 6000

        try:
            query = validate_text_length(
                query,
                field="query",
                max_chars=conf.TOOL_MAX_QUERY_CHARS,
                error_code="QUERY_TOO_LONG",
            )
            section = validate_text_length(
                section,
                field="section",
                max_chars=conf.TOOL_MAX_SECTION_CHARS,
                error_code="SECTION_TOO_LONG",
            )
            ids = parse_local_ids(paper_ids, max_ids=conf.TOOL_MAX_PAPER_IDS)
        except ToolSecurityError as exc:
            return _security_failure(exc, data=[])

        try:
            if ids:
                from rag.storage import get_chunks_for_paper_ids_readonly

                results = get_chunks_for_paper_ids_readonly(ids, max_chunks_per_paper=top_k, section=section)
                if not results:
                    pm = self._ensure_pm()
                    results = pm.get_chunks_for_paper_ids(ids, max_chunks_per_paper=top_k, section=section)
            else:
                if not query:
                    return ToolResult.fail(
                        "Either query or paper_ids must be provided.",
                        error_code="MISSING_RETRIEVAL_INPUT",
                        suggestion="Provide either a concise query or comma-separated paper_ids.",
                        data=[],
                )
                pm = self._ensure_pm()
                results = pm.search_knowledge(query, n_results=top_k)
                if section:
                    needle = section.lower()
                    results = [
                        res for res in results
                        if needle in str(res.get("section_name", "")).lower()
                        or needle in str(res.get("section_title", "")).lower()
                    ]
        except Exception as e:
            return ToolResult.fail(
                f"Evidence retrieval failed: {e}",
                error_code="EVIDENCE_RETRIEVAL_FAILED",
                suggestion="Check model availability, database health, and server logs.",
                data={"chunks": []},
            )

        if not results:
            target = ", ".join(ids) if ids else query
            return ToolResult.fail(
                f"No relevant evidence chunks found: {target}",
                error_code="LOCAL_CONTENT_NOT_FOUND",
                suggestion="Use list_local_database or search_local_database to confirm paper IDs and sections.",
                data={"chunks": []},
            )

        context, data = _format_evidence_chunks(results, max_chars=max_chars)
        return ToolResult.success(context, data=data)


class DbListTool(BaseTool):
    name = "list_local_database"
    description = "List all papers currently stored in the local database."
    params = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, paper_manager: Optional[PaperManager] = None, paper_db: Optional["PaperDB"] = None):
        self.pm = paper_manager
        self.db = paper_db

    def _ensure_pm(self):
        if self.pm is None and self.db is None:
            from rag.storage import PaperDB

            self.db = PaperDB()
        return self.pm, self.db

    def execute(self) -> ToolResult:
        try:
            pm, db = self._ensure_pm()
            papers = pm.list_all() if pm is not None else db.get_all_papers()
        except Exception:
            from rag.storage import list_papers_readonly

            papers = list_papers_readonly(limit=100)
        if not papers:
            return ToolResult.success("The local database is empty.", data=[])
        res = "=== Local Database Papers ===\n"
        for p in papers:
            sections = p.get("sections") or []
            if not sections and self.db is not None and hasattr(self.db, "get_paper_sections"):
                try:
                    sections = self.db.get_paper_sections(p["id"])
                except Exception:
                    sections = []
            section_text = ", ".join(sections[:8]) if sections else "No cached sections"
            res += f"- ID: {p['id']} | Title: {p['title']} | Sections: {section_text}\n"
        res, truncated = truncate_text(res, conf.TOOL_MAX_RETURN_CHARS)
        return ToolResult.success(
            res,
            data={"papers": papers, "truncated": truncated, "max_return_chars": conf.TOOL_MAX_RETURN_CHARS},
        )


class DbSearchTool(BaseTool):
    name = "search_local_database"
    description = (
        "Search local paper metadata records by title, author, abstract, or tags. "
        "Use this to find paper IDs and metadata in the local SQLite paper library; "
        "use retrieve_evidence_chunks when full-text semantic chunks are needed."
    )
    params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword, paper title fragment, author name, tag, or concise metadata search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matching paper records to return (1-50, default 20).",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, paper_db: Optional["PaperDB"] = None):
        self.db = paper_db

    def _ensure_db(self):
        if self.db is None:
            from rag.storage import PaperDB

            self.db = PaperDB()
        return self.db

    def execute(self, query: str, limit: int = 20) -> ToolResult:
        try:
            query = validate_text_length(
                query,
                field="query",
                max_chars=conf.TOOL_MAX_QUERY_CHARS,
                error_code="QUERY_TOO_LONG",
            )
        except ToolSecurityError as exc:
            return _security_failure(exc, data=[])
        if not query:
            return ToolResult.fail(
                "query must not be empty",
                error_code="EMPTY_QUERY",
                suggestion="Provide a title fragment, author, tag, or metadata keyword.",
                data=[],
            )
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 20

        try:
            db = self._ensure_db()
            papers = db.search_papers(query)[:limit]
        except Exception:
            from rag.storage import list_papers_readonly

            papers = list_papers_readonly(query=query, limit=limit)
        if not papers:
            return ToolResult.success(f"No local paper records matched query: {query}", data=[])

        lines = [f"=== Local Database Search Results ({len(papers)}) ==="]
        for p in papers:
            authors = (p.get("authors") or "").strip()
            year = p.get("publish_year") or "unknown year"
            tags = (p.get("tags") or "").strip()
            sections = p.get("sections") or []
            detail = f"- ID: {p['id']} | Title: {p['title']} | Year: {year}"
            if authors:
                detail += f" | Authors: {authors}"
            if tags:
                detail += f" | Tags: {tags}"
            if sections:
                detail += f" | Sections: {', '.join(sections[:8])}"
            lines.append(detail)
        text, truncated = truncate_text("\n".join(lines), conf.TOOL_MAX_RETURN_CHARS)
        return ToolResult.success(
            text,
            data={"papers": papers, "truncated": truncated, "max_return_chars": conf.TOOL_MAX_RETURN_CHARS},
        )


class PaperOutlineTool(BaseTool):
    name = "get_paper_outline"
    description = "Return a paper's section outline and cached-section metadata without returning body text."
    params = {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "Local paper ID, e.g. local_abc123."},
            "language": {
                "type": "string",
                "description": "Summary language used only for has_summary checks.",
                "enum": ["zh", "en"],
                "default": "en",
            },
            "detail_level": {
                "type": "string",
                "description": "Summary detail level used only for has_summary checks.",
                "enum": ["short", "medium", "long"],
                "default": "medium",
            },
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    }

    def __init__(self, paper_db: Optional["PaperDB"] = None):
        self.db = paper_db

    def _ensure_db(self):
        if self.db is None:
            from rag.storage import PaperDB

            self.db = PaperDB()
        return self.db

    def execute(self, paper_id: str, language: str = "en", detail_level: str = "medium") -> ToolResult:
        try:
            paper_id = validate_local_id(paper_id)
            language = _validate_language(language)
            detail_level = _validate_detail_level(detail_level)
        except ToolSecurityError as exc:
            return _security_failure(exc)

        outline = self._ensure_db().get_paper_outline(paper_id, language=language, detail_level=detail_level)
        if outline is None:
            return ToolResult.fail(
                f"Paper not found: {paper_id}",
                error_code="PAPER_NOT_FOUND",
                suggestion="Use list_local_database or search_local_database to find a valid paper_id.",
                data={"paper_id": paper_id},
            )

        lines = [f"Paper: {outline['title']} ({paper_id})", "Sections:"]
        for section in outline["sections"]:
            marker = "summary ready" if section["has_summary"] else "summary missing"
            lines.append(
                f"- {section['section']} | chunks={section['chunk_count']} "
                f"| chars={section['char_count']} | {marker}"
            )
        return ToolResult.success("\n".join(lines), data=outline)


class PaperProfileTool(BaseTool):
    name = "get_paper_profile"
    description = (
        "Return a compact cached paper profile for relevance checks, planning, and multi-paper comparison. "
        "This reads precomputed summary cache only; call build_paper_summary if not ready."
    )
    params = {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "Local paper ID, e.g. local_abc123."},
            "language": {
                "type": "string",
                "description": "Profile language.",
                "enum": ["zh", "en"],
                "default": "en",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum response characters (1000-12000, default 4000).",
                "minimum": 1000,
                "maximum": 12000,
                "default": 4000,
            },
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    }

    def __init__(self, paper_db: Optional["PaperDB"] = None):
        self.db = paper_db

    def _ensure_db(self):
        if self.db is None:
            from rag.storage import PaperDB

            self.db = PaperDB()
        return self.db

    def execute(self, paper_id: str, language: str = "en", max_chars: int = 4000) -> ToolResult:
        try:
            paper_id = validate_local_id(paper_id)
            language = _validate_language(language)
            max_chars = max(1000, min(int(max_chars), 12000))
        except (TypeError, ValueError):
            max_chars = 4000
        except ToolSecurityError as exc:
            return _security_failure(exc)

        profile = self._ensure_db().get_profile(paper_id, language)
        if not profile or profile.get("status") != "ready":
            return _summary_not_ready(paper_id)

        text, truncated = truncate_text(json.dumps(profile, ensure_ascii=False, indent=2), max_chars)
        data = {**profile, "truncated": truncated, "max_chars": max_chars}
        return ToolResult.success(text, data=data)


class PaperSummaryTool(BaseTool):
    name = "get_paper_summary"
    description = (
        "Return a cached structured paper summary for report drafting or method overview. "
        "This tool never generates a long LLM summary synchronously; call build_paper_summary if not ready. "
        "Use retrieve_evidence_chunks for exact supporting passages and citations."
    )
    params = {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "Local paper ID, e.g. local_abc123."},
            "detail_level": {
                "type": "string",
                "description": "Summary size.",
                "enum": ["short", "medium", "long"],
                "default": "medium",
            },
            "language": {
                "type": "string",
                "description": "Summary language.",
                "enum": ["zh", "en"],
                "default": "en",
            },
            "include_sections": {
                "type": "boolean",
                "description": "Whether to include cached section summaries.",
                "default": True,
            },
            "include_evidence": {
                "type": "boolean",
                "description": "Reserved for future citation snippets. Use retrieve_evidence_chunks for evidence.",
                "default": False,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum response characters (1000-50000, default 12000).",
                "minimum": 1000,
                "maximum": 50000,
                "default": 12000,
            },
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    }

    def __init__(self, paper_db: Optional["PaperDB"] = None):
        self.db = paper_db

    def _ensure_db(self):
        if self.db is None:
            from rag.storage import PaperDB

            self.db = PaperDB()
        return self.db

    def execute(
        self,
        paper_id: str,
        detail_level: str = "medium",
        language: str = "en",
        include_sections: bool = True,
        include_evidence: bool = False,
        max_chars: int = 12000,
    ) -> ToolResult:
        try:
            paper_id = validate_local_id(paper_id)
            language = _validate_language(language)
            detail_level = _validate_detail_level(detail_level)
            max_chars = max(1000, min(int(max_chars), 50000))
        except (TypeError, ValueError):
            max_chars = 12000
        except ToolSecurityError as exc:
            return _security_failure(exc)

        db = self._ensure_db()
        summary = db.get_summary(paper_id, language, detail_level)
        if not summary or summary.get("summary_status") != "ready":
            return _summary_not_ready(paper_id)

        section_summaries = []
        if include_sections:
            section_summaries = db.get_section_summaries(paper_id, language, detail_level)
        payload = {
            **summary,
            "section_summaries": section_summaries,
            "include_evidence": bool(include_evidence),
            "evidence_note": "Use retrieve_evidence_chunks for exact citations." if include_evidence else "",
        }
        text, truncated = truncate_text(json.dumps(payload, ensure_ascii=False, indent=2), max_chars)
        payload["truncated"] = truncated
        payload["max_chars"] = max_chars
        return ToolResult.success(text, data=payload)


class BuildPaperSummaryTool(BaseTool):
    name = "build_paper_summary"
    description = (
        "Submit a background job to build or refresh cached paper profile, paper summaries, "
        "and section summaries with the configured local summary model; "
        "poll get_tool_job_status for completion."
    )
    params = {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "Local paper ID, e.g. local_abc123."},
            "language": {
                "type": "string",
                "description": "Summary language to build: zh, en, or both.",
                "enum": ["zh", "en", "both"],
                "default": "en",
            },
            "detail_levels": {
                "type": "array",
                "items": {"type": "string", "enum": ["short", "medium", "long"]},
                "description": "Detail levels to build. Defaults to all levels.",
                "default": ["short", "medium", "long"],
            },
            "force_rebuild": {
                "type": "boolean",
                "description": "Rebuild even when cache source_hash/model/prompt match.",
                "default": False,
            },
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    }

    def __init__(self, paper_db: Optional["PaperDB"] = None):
        self.db = paper_db

    def _ensure_db(self):
        if self.db is None:
            from rag.storage import PaperDB

            self.db = PaperDB()
        return self.db

    def execute(
        self,
        paper_id: str,
        language: str = "en",
        detail_levels: Any = None,
        force_rebuild: bool = False,
    ) -> ToolResult:
        try:
            paper_id = validate_local_id(paper_id)
            language = _validate_language(language, allow_both=True)
            levels = _parse_detail_levels(detail_levels)
        except ToolSecurityError as exc:
            return _security_failure(exc)

        job = summary_job_manager.submit(
            "build_paper_summary",
            {
                "paper_id": paper_id,
                "language": language,
                "detail_levels": levels,
                "force_rebuild": bool(force_rebuild),
            },
            lambda: self._execute_build(paper_id, language, levels, bool(force_rebuild)),
        )
        msg = (
            f"Summary build job submitted. job_id={job['job_id']} status={job['status']}. "
            "Use get_tool_job_status with this job_id to check progress."
        )
        return ToolResult.success(msg, data=job)

    def _execute_build(
        self,
        paper_id: str,
        language: str,
        detail_levels: list[str],
        force_rebuild: bool,
    ) -> ToolResult:
        db = self._ensure_db()
        try:
            from rag.plugins.summary_model import get_summary_model_manager

            summary_generator = get_summary_model_manager()
        except Exception as exc:
            return ToolResult.fail(
                f"Summary model unavailable: {exc}",
                error_code="SUMMARY_MODEL_UNAVAILABLE",
                suggestion=(
                    "Check vLLM installation, RAG_SUMMARY_MODEL_PATH, RAG_SUMMARY_MODEL_REPO, "
                    "RAG_SUMMARY_AUTO_DOWNLOAD, and RAG_SUMMARY_OFFLINE_MODE."
                ),
                data={"paper_id": paper_id},
            )
        languages = ["zh", "en"] if language == "both" else [language]
        reports = []
        for item in languages:
            report = db.build_summary_cache(
                paper_id,
                language=item,
                detail_levels=detail_levels,
                force_rebuild=force_rebuild,
                summary_generator=summary_generator,
            )
            reports.append(report)
        failures = [report for report in reports if report.get("status") != "ready"]
        if failures:
            code = str(failures[0].get("error_code") or "SUMMARY_GENERATION_FAILED")
            return ToolResult.fail(
                f"Summary build failed for {paper_id}: {code}",
                error_code=code,
                suggestion="Confirm the paper exists and has cached chunks; re-ingest the PDF if needed.",
                data={"paper_id": paper_id, "reports": reports},
            )
        lines = [f"Summary cache ready for {paper_id}."]
        for report in reports:
            lines.append(
                f"- {report['language']}: built={report['detail_levels_built']} "
                f"skipped={report['detail_levels_skipped']} model={report.get('model_name')}"
            )
        return ToolResult.success("\n".join(lines), data={"paper_id": paper_id, "reports": reports})


def _submit_summary_build_job(
    *,
    paper_id: str,
    language: str,
    detail_levels: list[str],
    force_rebuild: bool = False,
) -> dict:
    return summary_job_manager.submit(
        "build_paper_summary",
        {
            "paper_id": paper_id,
            "language": language,
            "detail_levels": detail_levels,
            "force_rebuild": bool(force_rebuild),
        },
        lambda: BuildPaperSummaryTool()._execute_build(paper_id, language, detail_levels, bool(force_rebuild)),
    )


class DbAddTool(BaseTool):
    name = "add_paper_to_database"
    description = (
        "Submit a background job to parse a PDF file and import it into the local database and vector store. "
        "Returns a job_id immediately; call get_tool_job_status to check progress and final result. "
        "When a duplicate is detected (via content hash, DOI, or normalized title), "
        "the on_duplicate policy controls what happens: 'skip' (default) stops without "
        "importing, 'replace' removes the existing entry and imports the new one, "
        "'keep_both' imports regardless alongside the existing entry."
    )
    params = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Full filename including .pdf extension."},
            "on_duplicate": {
                "type": "string",
                "description": "Duplicate resolution policy: skip, replace, or keep_both (default skip).",
                "enum": ["skip", "replace", "keep_both"],
                "default": "skip",
            },
            "build_summary": {
                "type": "boolean",
                "description": "Submit a separate build_paper_summary job after successful import. Default true.",
                "default": True,
            },
            "summary_language": {
                "type": "string",
                "description": "Summary language for the optional summary job: zh, en, or both.",
                "enum": ["zh", "en", "both"],
                "default": "en",
            },
            "summary_detail_levels": {
                "type": "array",
                "items": {"type": "string", "enum": ["short", "medium", "long"]},
                "description": "Detail levels for optional summary job. Default ['short'].",
                "default": ["short"],
            },
        },
        "required": ["filename"],
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager()
        return self.pm

    def execute(
        self,
        filename: str,
        on_duplicate: str = "skip",
        build_summary: bool = True,
        summary_language: str = "en",
        summary_detail_levels: Any = None,
    ) -> ToolResult:
        logger.info(
            "[tool] add_paper_to_database filename=%s on_duplicate=%s build_summary=%s",
            filename,
            on_duplicate,
            build_summary,
        )
        try:
            pdf_path = resolve_safe_pdf_filename(filename, conf.PAPERS_DIR)
        except ToolSecurityError as exc:
            return _security_failure(exc)
        build_summary_value, bool_error = _coerce_tool_bool(build_summary, field="build_summary")
        if bool_error is not None:
            return bool_error
        try:
            summary_language = _validate_language(summary_language, allow_both=True)
            summary_levels = _parse_detail_levels(summary_detail_levels or ["short"])
        except ToolSecurityError as exc:
            return _security_failure(exc)
        if on_duplicate not in ("skip", "replace", "keep_both"):
            return ToolResult.fail(
                f"Invalid on_duplicate: {on_duplicate!r}",
                error_code="INVALID_DUPLICATE_POLICY",
                suggestion="Use one of: skip, replace, keep_both.",
            )

        os.makedirs(conf.PAPERS_DIR, exist_ok=True)
        if not pdf_path.is_file():
            return ToolResult.fail(
                f"File not found: {pdf_path}",
                error_code="PDF_NOT_FOUND",
                suggestion="Copy the PDF into PAPERS_DIR and retry with the exact filename.",
            )

        job = job_manager.submit(
            "add_paper_to_database",
            {
                "filename": pdf_path.name,
                "on_duplicate": on_duplicate,
                "build_summary": build_summary_value,
                "summary_language": summary_language,
                "summary_detail_levels": summary_levels,
            },
            lambda: self._execute_ingest(
                pdf_path.name,
                on_duplicate,
                build_summary_value,
                summary_language,
                summary_levels,
            ),
        )
        msg = (
            f"Import job submitted. job_id={job['job_id']} status={job['status']}. "
            "Use get_tool_job_status with this job_id to check progress."
        )
        return ToolResult.success(msg, data=job)

    def _execute_ingest(
        self,
        filename: str,
        on_duplicate: str,
        build_summary: bool,
        summary_language: str,
        summary_detail_levels: list[str],
    ) -> ToolResult:
        pm = self._ensure_pm()
        try:
            pdf_path = resolve_safe_pdf_filename(filename, conf.PAPERS_DIR)
        except ToolSecurityError as exc:
            return _security_failure(exc)
        try:
            success, msg = pm.ingest_pdf(str(pdf_path), tags="Agent-Added", on_duplicate=on_duplicate)
        except Exception as exc:
            logger.exception(
                "[tool] add_paper_to_database failed filename=%s on_duplicate=%s error=%s",
                filename,
                on_duplicate,
                exc,
            )
            return ToolResult.fail(
                f"Ingestion failed: {exc}",
                error_code="PDF_INGESTION_EXCEPTION",
                suggestion="Check model availability, PDF parser dependencies, and server logs.",
            )
        if not success:
            return ToolResult.fail(
                msg,
                error_code=_classify_ingest_failure(msg),
                suggestion=_suggest_for_ingest_failure(msg),
            )
        paper_id = str(getattr(pm, "last_ingested_paper_id", "") or "")
        data: dict[str, Any] = {"paper_id": paper_id, "summary_job": None}
        if build_summary and data["paper_id"]:
            summary_job = _submit_summary_build_job(
                paper_id=data["paper_id"],
                language=summary_language,
                detail_levels=summary_detail_levels,
            )
            data["summary_job"] = summary_job
            msg = f"{msg}\nSummary build job submitted. summary_job_id={summary_job['job_id']}."
        return ToolResult.success(msg, data=data)


class DbImportDirectoryTool(BaseTool):
    name = "import_papers_from_directory"
    description = (
        "Submit a background job to import every PDF in PAPERS_DIR or one of its child directories. "
        "The subdir parameter must be relative to PAPERS_DIR; absolute paths and traversal are rejected. "
        "The job imports files one by one, preserves per-file success/skip/failure results, and continues "
        "when an individual PDF fails. Use get_tool_job_status to check progress and final metrics."
    )
    params = {
        "type": "object",
        "properties": {
            "subdir": {
                "type": "string",
                "description": "Optional relative directory under PAPERS_DIR. Empty means PAPERS_DIR itself.",
                "default": "",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to include PDFs in nested child directories. Default false.",
                "default": False,
            },
            "on_duplicate": {
                "type": "string",
                "description": "Duplicate resolution policy: skip, replace, or keep_both (default skip).",
                "enum": ["skip", "replace", "keep_both"],
                "default": "skip",
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum number of PDFs to process in one job, between 1 and 1000. Default 200.",
                "minimum": 1,
                "maximum": 1000,
                "default": 200,
            },
            "dry_run": {
                "type": "boolean",
                "description": "Only list matching PDFs without importing them. Default false.",
                "default": False,
            },
            "build_summary": {
                "type": "boolean",
                "description": "Submit separate build_paper_summary jobs for successfully imported PDFs. Default true.",
                "default": True,
            },
            "summary_language": {
                "type": "string",
                "description": "Summary language for optional summary jobs: zh, en, or both.",
                "enum": ["zh", "en", "both"],
                "default": "en",
            },
            "summary_detail_levels": {
                "type": "array",
                "items": {"type": "string", "enum": ["short", "medium", "long"]},
                "description": "Detail levels for optional summary jobs. Default ['short'].",
                "default": ["short"],
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager()
        return self.pm

    def execute(
        self,
        subdir: str = "",
        recursive: bool = False,
        on_duplicate: str = "skip",
        max_files: int = 200,
        dry_run: bool = False,
        build_summary: bool = True,
        summary_language: str = "en",
        summary_detail_levels: Any = None,
    ) -> ToolResult:
        logger.info(
            "[tool] import_papers_from_directory subdir=%s recursive=%s on_duplicate=%s "
            "max_files=%s dry_run=%s build_summary=%s",
            subdir,
            recursive,
            on_duplicate,
            max_files,
            dry_run,
            build_summary,
        )
        try:
            directory = resolve_safe_papers_subdir(subdir, conf.PAPERS_DIR)
        except ToolSecurityError as exc:
            return _security_failure(exc)
        recursive_value, bool_error = _coerce_tool_bool(recursive, field="recursive")
        if bool_error is not None:
            return bool_error
        dry_run_value, bool_error = _coerce_tool_bool(dry_run, field="dry_run")
        if bool_error is not None:
            return bool_error
        build_summary_value, bool_error = _coerce_tool_bool(build_summary, field="build_summary")
        if bool_error is not None:
            return bool_error
        try:
            summary_language = _validate_language(summary_language, allow_both=True)
            summary_levels = _parse_detail_levels(summary_detail_levels or ["short"])
        except ToolSecurityError as exc:
            return _security_failure(exc)
        if on_duplicate not in ("skip", "replace", "keep_both"):
            return ToolResult.fail(
                f"Invalid on_duplicate: {on_duplicate!r}",
                error_code="INVALID_DUPLICATE_POLICY",
                suggestion="Use one of: skip, replace, keep_both.",
            )
        try:
            limit = int(max_files)
        except (TypeError, ValueError):
            return ToolResult.fail(
                f"Invalid max_files: {max_files!r}",
                error_code="INVALID_MAX_FILES",
                suggestion="Use an integer between 1 and 1000.",
            )
        if limit < 1 or limit > 1000:
            return ToolResult.fail(
                f"Invalid max_files: {limit}",
                error_code="INVALID_MAX_FILES",
                suggestion="Use an integer between 1 and 1000.",
            )
        if not directory.exists():
            return ToolResult.fail(
                f"Directory not found: {directory}",
                error_code="DIRECTORY_NOT_FOUND",
                suggestion="Create the directory under PAPERS_DIR or use an existing subdir.",
            )
        if not directory.is_dir():
            return ToolResult.fail(
                f"Path is not a directory: {directory}",
                error_code="DIRECTORY_REQUIRED",
                suggestion="Pass a directory under PAPERS_DIR, not a file.",
            )

        relative = "."
        try:
            relative = directory.relative_to(Path(conf.PAPERS_DIR).expanduser().resolve()).as_posix() or "."
        except ValueError:
            relative = directory.name

        job = job_manager.submit(
            "import_papers_from_directory",
            {
                "subdir": relative,
                "recursive": recursive_value,
                "on_duplicate": on_duplicate,
                "max_files": limit,
                "dry_run": dry_run_value,
                "build_summary": build_summary_value,
                "summary_language": summary_language,
                "summary_detail_levels": summary_levels,
            },
            lambda: self._execute_import_directory(
                str(directory),
                recursive_value,
                on_duplicate,
                limit,
                dry_run_value,
                build_summary_value,
                summary_language,
                summary_levels,
            ),
        )
        msg = (
            f"Batch import job submitted. job_id={job['job_id']} status={job['status']}. "
            "Use get_tool_job_status with this job_id to check progress."
        )
        return ToolResult.success(msg, data=job)

    def _execute_import_directory(
        self,
        directory: str,
        recursive: bool,
        on_duplicate: str,
        max_files: int,
        dry_run: bool,
        build_summary: bool,
        summary_language: str,
        summary_detail_levels: list[str],
    ) -> ToolResult:
        pm = self._ensure_pm()
        try:
            report = pm.ingest_directory(
                directory,
                recursive=recursive,
                on_duplicate=on_duplicate,
                max_files=max_files,
                dry_run=dry_run,
                tags="Agent-Added",
            )
        except Exception as exc:
            logger.exception("[tool] import_papers_from_directory failed directory=%s error=%s", directory, exc)
            return ToolResult.fail(
                f"Batch import failed: {exc}",
                error_code="BATCH_IMPORT_EXCEPTION",
                suggestion="Check PDF parser dependencies, model availability, directory permissions, and server logs.",
            )
        if report.get("status") != "succeeded":
            return ToolResult.fail(
                str(report.get("message") or "Batch import failed."),
                error_code=str(report.get("error_code") or "BATCH_IMPORT_FAILED"),
                suggestion="Check the directory path and import parameters.",
                data=report,
            )
        summary_jobs = []
        if build_summary and not dry_run:
            for item in report.get("results", []):
                if item.get("status") != "imported" or not item.get("paper_id"):
                    continue
                summary_job = _submit_summary_build_job(
                    paper_id=str(item["paper_id"]),
                    language=summary_language,
                    detail_levels=summary_detail_levels,
                )
                item["summary_job_id"] = summary_job["job_id"]
                summary_jobs.append(summary_job)
            report["summary_jobs"] = summary_jobs
        lines = [
            str(report.get("message") or "Batch import completed."),
            f"directory={report.get('directory')}",
            (
                f"total_found={report.get('total_found')} processed={report.get('processed')} "
                f"imported={report.get('imported')} skipped={report.get('skipped')} failed={report.get('failed')}"
            ),
        ]
        if build_summary:
            lines.append(f"summary_jobs_submitted={len(summary_jobs)}")
        if report.get("limited"):
            lines.append(f"Result limited by max_files={report.get('max_files')}.")
        return ToolResult.success("\n".join(lines), data=report)


class ToolJobStatusTool(BaseTool):
    name = "get_tool_job_status"
    description = (
        "Long-poll status and final result for a background tool job, such as add_paper_to_database. "
        "For queued/running jobs this tool waits with an arithmetic backoff before returning, "
        "so clients should call it directly instead of rapid manual polling."
    )
    params = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Job ID returned by a background tool."}
        },
        "required": ["job_id"],
        "additionalProperties": False,
    }

    def execute(self, job_id: str) -> ToolResult:
        try:
            job_id = validate_job_id(job_id)
        except ToolSecurityError as exc:
            return _security_failure(exc, data={"job_id": str(job_id or "").strip()})
        record = _observe_tool_job_record(job_id)
        if record is None:
            return ToolResult.fail(
                f"Job not found: {job_id}",
                error_code="JOB_NOT_FOUND",
                suggestion="Use the exact job_id returned by the long-running tool.",
                data={"job_id": job_id},
            )
        record, waited_seconds, next_wait_seconds = _wait_for_nonterminal_job(job_id, record)
        record["wait_seconds_applied"] = waited_seconds
        record["next_wait_seconds"] = next_wait_seconds
        lines = [
            f"Job ID: {record['job_id']}",
            f"Type: {record['job_type']}",
            f"Queue: {record.get('queue', 'default')}",
            f"Status: {record['status']}",
            f"Status checks: {record.get('status_checks', 0)}",
            f"Waited seconds: {waited_seconds:.2f}",
            f"Created: {record['created_at']}",
        ]
        if next_wait_seconds > 0:
            lines.append(f"Next suggested wait seconds: {next_wait_seconds:.2f}")
        if record.get("started_at"):
            lines.append(f"Started: {record['started_at']}")
        if record.get("finished_at"):
            lines.append(f"Finished: {record['finished_at']}")
        if record.get("result"):
            lines.append(f"Result: {record['result'].get('result')}")
        return ToolResult.success("\n".join(lines), data=record)


class RagHealthCheckTool(BaseTool):
    name = "rag_health_check"
    description = "Return lightweight health information for RAG storage paths, model paths, and background jobs."
    params = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self) -> ToolResult:
        from pathlib import Path

        checks = {
            "db_dir_exists": Path(conf.DB_DIR).exists(),
            "papers_dir_exists": Path(conf.PAPERS_DIR).exists(),
            "models_dir_exists": Path(conf.MODELS_DIR).exists(),
            "bge_m3_path_exists": Path(conf.BGE_M3_MODEL_PATH).exists(),
            "bge_reranker_path_exists": Path(conf.BGE_RERANKER_MODEL_PATH).exists(),
        }
        try:
            from rag.storage import model_manager

            summary_model = model_manager.get_summary_model_status()
        except Exception as exc:
            summary_model = {"error": str(exc)}
        jobs = _combined_job_stats()
        lines = ["=== RAG Health Check ==="]
        for key, value in checks.items():
            lines.append(f"- {key}: {value}")
        if "complete" in summary_model:
            lines.append(f"- summary_model_complete: {summary_model['complete']}")
            lines.append(f"- summary_model_path: {summary_model['path']}")
            lines.append(f"- summary_model_backend: {summary_model.get('backend')}")
            if summary_model.get("backend") == "api":
                lines.append(f"- summary_model_api_base_url: {summary_model.get('api_base_url')}")
                lines.append(f"- summary_model_api_model: {summary_model.get('api_model')}")
                lines.append(f"- summary_model_api_ping_attempted: {summary_model.get('api_ping_attempted')}")
                lines.append(f"- summary_model_api_ping_ok: {summary_model.get('api_ping_ok')}")
                lines.append(f"- summary_model_api_ping_status_code: {summary_model.get('api_ping_status_code')}")
                lines.append(f"- summary_model_api_ping_latency_ms: {summary_model.get('api_ping_latency_ms')}")
                lines.append(f"- summary_model_api_ping_error: {summary_model.get('api_ping_error')}")
            lines.append(f"- summary_model_auto_download: {summary_model.get('auto_download')}")
            lines.append(f"- summary_model_offline_mode: {summary_model.get('offline_mode')}")
            lines.append(f"- summary_model_python: {summary_model.get('python_executable')}")
            lines.append(f"- summary_model_platform: {summary_model.get('platform')}")
            lines.append(f"- summary_model_huggingface_hub_installed: {summary_model.get('huggingface_hub_installed')}")
            lines.append(f"- summary_model_vllm_installed: {summary_model.get('vllm_installed')}")
            lines.append(f"- summary_model_download_trigger: {summary_model.get('download_trigger')}")
            lines.append(f"- summary_model_model_download_available: {summary_model.get('model_download_available')}")
            lines.append(f"- summary_model_download_would_start: {summary_model.get('download_would_start')}")
            lines.append(f"- summary_model_generation_ready: {summary_model.get('generation_ready')}")
            blocker = summary_model.get("generation_blocker") or summary_model.get("download_blocker")
            lines.append(f"- summary_model_blocker: {blocker}")
            lines.append(f"- summary_model_next_action: {summary_model.get('next_action')}")
        lines.append(f"- jobs_total: {jobs['total']}")
        lines.append(f"- jobs_queued: {jobs['queued']}")
        lines.append(f"- jobs_running: {jobs['running']}")
        lines.append(f"- summary_jobs_queued: {jobs['queues']['summary']['queued']}")
        lines.append(f"- summary_jobs_running: {jobs['queues']['summary']['running']}")
        lines.append(f"- summary_job_max_workers: {jobs['queues']['summary']['max_workers']}")
        return ToolResult.success(
            "\n".join(lines),
            data={"checks": checks, "summary_model": summary_model, "jobs": jobs},
        )


class DbDeleteTool(BaseTool):
    name = "delete_paper_from_database"
    description = "Delete a paper from the local database (both SQLite and vector store)."
    params = {
        "type": "object",
        "properties": {
            "local_id": {"type": "string", "description": "The local paper ID to delete."}
        },
        "required": ["local_id"],
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager()
        return self.pm

    def execute(self, local_id: str) -> ToolResult:
        logger.info(f"[tool] delete_paper local_id={local_id}")
        try:
            local_id = validate_local_id(local_id)
        except ToolSecurityError as exc:
            return _security_failure(exc)
        pm = self._ensure_pm()
        status, msg = pm.delete_paper(local_id)
        if not status:
            return ToolResult.fail(
                msg,
                error_code="DELETE_PAPER_FAILED",
                suggestion="Verify the local_id exists in list_local_database.",
            )
        return ToolResult.success(msg)


class DedupDatabaseTool(BaseTool):
    name = "dedup_local_database"
    description = (
        "Scan the local database for duplicate papers using SHA256 hash, DOI, or "
        "normalized title matching. 'report' (default) lists duplicate groups without "
        "changes. 'prune' keeps the newest entry per group and deletes the rest."
    )
    params = {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "description": "Dedup strategy: 'report' lists groups without changes, 'prune' deletes duplicates.",
                "enum": ["report", "prune"],
                "default": "report",
            },
            "backfill": {
                "type": "boolean",
                "description": "Whether to backfill legacy rows first (default true).",
                "default": True,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager()
        return self.pm

    def execute(self, strategy: str = "report", backfill: bool = True) -> ToolResult:
        pm = self._ensure_pm()
        logger.info(f"[tool] dedup_local_database strategy={strategy} backfill={backfill}")
        if strategy not in ("report", "prune"):
            return ToolResult.fail(
                f"Invalid strategy: {strategy!r}. Use 'report' or 'prune'.",
                error_code="INVALID_DEDUP_STRATEGY",
                suggestion="Use strategy='report' or strategy='prune'.",
            )
        try:
            report = pm.deduplicate_database(strategy=strategy, backfill=bool(backfill))
        except Exception as e:
            return ToolResult.fail(
                f"Dedup scan failed: {e}",
                error_code="DEDUP_SCAN_FAILED",
                suggestion="Check database health and server logs.",
            )

        groups = report["groups"]
        pruned = report["pruned"]
        total = report["total_papers"]

        if not groups:
            return ToolResult.success(
                f"No duplicate papers found across {total} total papers. Database is clean.",
            )

        lines = [
            f"Dedup scan complete: {len(groups)} duplicate group(s) found "
            f"across {total} total papers.",
            "",
        ]
        for idx, grp in enumerate(groups):
            reasons = ", ".join(grp["reasons"]) or "unknown"
            members = grp["members"]
            lines.append(f"--- Group {idx + 1} (matched by: {reasons}) ---")
            for m in members:
                title = (m.get("title") or "<untitled>")[:80]
                lines.append(f"    ID: {m['id']} | {title}")
            lines.append("")

        if strategy == "prune" and pruned:
            lines.append(f"Pruned {len(pruned)} duplicate paper(s).")
            lines.append(f"Remaining papers: {total - len(pruned)}")

        return ToolResult.success("\n".join(lines))


class BackfillMetadataTool(BaseTool):
    name = "backfill_paper_metadata"
    description = (
        "Recompute dedup metadata (content hash, normalized title) for papers "
        "ingested before dedup columns existed. Required before dedup if legacy "
        "papers exist in the database."
    )
    params = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager()
        return self.pm

    def execute(self) -> ToolResult:
        pm = self._ensure_pm()
        logger.info("[tool] backfill_paper_metadata")
        try:
            report = pm.backfill_metadata()
        except Exception as e:
            return ToolResult.fail(
                f"Backfill failed: {e}",
                error_code="BACKFILL_FAILED",
                suggestion="Check database health, PDF paths, and server logs.",
            )

        scanned = report.get("scanned", 0)
        updated = report.get("updated", 0)
        missing = report.get("missing_files", 0)

        msg = (
            f"Backfill complete: scanned {scanned} paper(s), "
            f"updated {updated} with dedup metadata."
        )
        if missing:
            msg += f" {missing} paper(s) have missing files and could not be hashed."
        return ToolResult.success(msg)


class RetrievalQualityEvalTool(BaseTool):
    name = "evaluate_retrieval_quality"
    description = (
        "Submit a background retrieval-quality evaluation job from a JSONL dataset. "
        "Each row must include query and at least one expected_paper_ids, expected_chunk_ids, "
        "or expected_sections field. "
        "The job reports recall, precision, MRR, and metric deltas across retrieval modes."
    )
    params = {
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Project-relative JSONL dataset path. Defaults to RETRIEVAL_EVAL_DATASET_PATH.",
                "default": "",
            },
            "modes": {
                "type": "string",
                "description": "Comma-separated retrieval modes to compare: hybrid, dense, bm25.",
                "default": "hybrid,dense,bm25",
            },
            "top_k": {
                "type": "integer",
                "description": "Largest k for evaluation metrics (1-20, default 10).",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
            "max_cases": {
                "type": "integer",
                "description": "Maximum dataset rows to evaluate. Defaults to RETRIEVAL_EVAL_MAX_CASES.",
                "minimum": 1,
                "default": 0,
            },
            "use_hyde": {
                "type": "boolean",
                "description": "Whether to enable HyDE expansion during evaluation.",
                "default": False,
            },
            "rerank": {
                "type": "boolean",
                "description": "Whether to apply the BGE reranker during evaluation.",
                "default": True,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            self.pm = PaperManager(enable_hyde=False)
        return self.pm

    def execute(
        self,
        dataset_path: str = "",
        modes: str = "hybrid,dense,bm25",
        top_k: int = 10,
        max_cases: int = 0,
        use_hyde: bool = False,
        rerank: bool = True,
    ) -> ToolResult:
        try:
            dataset = resolve_project_file(
                dataset_path or conf.RETRIEVAL_EVAL_DATASET_PATH,
                conf.PROJECT_ROOT,
                suffix=".jsonl",
            )
        except ToolSecurityError as exc:
            return _security_failure(exc)
        if not dataset.is_file():
            return ToolResult.fail(
                f"Evaluation dataset not found: {dataset}",
                error_code="EVAL_DATASET_NOT_FOUND",
                suggestion="Create a JSONL dataset inside the project or set RETRIEVAL_EVAL_DATASET_PATH.",
            )
        try:
            top_k = max(1, min(int(top_k), 20))
        except (TypeError, ValueError):
            top_k = 10
        try:
            max_cases = int(max_cases)
        except (TypeError, ValueError):
            max_cases = 0
        if max_cases <= 0:
            max_cases = conf.RETRIEVAL_EVAL_MAX_CASES
        max_cases = max(1, min(max_cases, conf.RETRIEVAL_EVAL_MAX_CASES))

        selected_modes = [item.strip().lower() for item in str(modes or "").split(",") if item.strip()]
        if not selected_modes:
            selected_modes = ["hybrid", "dense", "bm25"]
        invalid = [mode for mode in selected_modes if mode not in {"hybrid", "dense", "bm25"}]
        if invalid:
            return ToolResult.fail(
                f"Unsupported retrieval mode(s): {', '.join(invalid)}",
                error_code="INVALID_RETRIEVAL_MODE",
                suggestion="Use modes from: hybrid, dense, bm25.",
            )

        job = job_manager.submit(
            "evaluate_retrieval_quality",
            {
                "dataset_path": str(dataset),
                "modes": selected_modes,
                "top_k": top_k,
                "max_cases": max_cases,
                "use_hyde": bool(use_hyde),
                "rerank": bool(rerank),
            },
            lambda: self._execute_eval(dataset, selected_modes, top_k, max_cases, bool(use_hyde), bool(rerank)),
        )
        msg = (
            f"Retrieval evaluation job submitted. job_id={job['job_id']} status={job['status']}. "
            "Use get_tool_job_status with this job_id to check progress and final metrics."
        )
        return ToolResult.success(msg, data=job)

    def _execute_eval(
        self,
        dataset: Path,
        modes: list[str],
        top_k: int,
        max_cases: int,
        use_hyde: bool,
        rerank: bool,
    ) -> ToolResult:
        from rag.evaluation import compare_reports, evaluate_retrieval_cases, load_retrieval_cases_jsonl

        try:
            cases = load_retrieval_cases_jsonl(dataset, max_cases=max_cases)
        except Exception as exc:
            return ToolResult.fail(
                f"Failed to load retrieval evaluation dataset: {exc}",
                error_code="EVAL_DATASET_INVALID",
                suggestion="Check JSONL rows include query and expected_* fields.",
            )
        if not cases:
            return ToolResult.fail(
                f"Evaluation dataset is empty: {dataset}",
                error_code="EVAL_DATASET_EMPTY",
                suggestion="Add JSONL rows with query and expected labels.",
            )

        pm = self._ensure_pm()
        k_values = [k for k in (1, 3, 5, 10, top_k) if k <= top_k]
        reports = {}
        for mode in modes:
            def search_fn(query: str, k: int, mode: str = mode) -> list[dict]:
                return pm.search_knowledge(
                    query,
                    n_results=k,
                    use_hyde=use_hyde,
                    retrieval_mode=mode,
                    rerank=rerank,
                )

            reports[mode] = evaluate_retrieval_cases(
                cases,
                search_fn,
                k_values=k_values,
            )

        baseline_mode = modes[0]
        diffs = {
            mode: compare_reports(reports[baseline_mode], report)
            for mode, report in reports.items()
            if mode != baseline_mode
        }
        lines = [
            "=== Retrieval Quality Evaluation ===",
            f"Dataset: {dataset}",
            f"Cases: {len(cases)}",
            f"Baseline: {baseline_mode}",
        ]
        for mode, report in reports.items():
            agg = report["aggregate"]
            lines.append(
                f"- {mode}: mrr={agg.get('mrr', 0.0):.4f}, "
                f"paper_recall@{top_k}={agg.get(f'paper_recall@{top_k}', 0.0):.4f}, "
                f"chunk_recall@{top_k}={agg.get(f'chunk_recall@{top_k}', 0.0):.4f}, "
                f"precision@{top_k}={agg.get(f'precision@{top_k}', 0.0):.4f}"
            )
        for mode, diff in diffs.items():
            regressions = diff.get("regressions") or {}
            lines.append(f"- delta vs {baseline_mode} for {mode}: regressions={len(regressions)}")

        report_path = self._write_eval_report(
            dataset=dataset,
            modes=modes,
            top_k=top_k,
            max_cases=max_cases,
            use_hyde=use_hyde,
            rerank=rerank,
            reports=reports,
            diffs=diffs,
            baseline_mode=baseline_mode,
        )
        lines.append(f"Report: {report_path}")
        text, truncated = truncate_text("\n".join(lines), conf.TOOL_MAX_RETURN_CHARS)
        return ToolResult.success(
            text,
            data={
                "dataset_path": str(dataset),
                "report_path": str(report_path),
                "baseline_mode": baseline_mode,
                "reports": reports,
                "diffs": diffs,
                "truncated": truncated,
            },
        )

    @staticmethod
    def _write_eval_report(
        *,
        dataset: Path,
        modes: list[str],
        top_k: int,
        max_cases: int,
        use_hyde: bool,
        rerank: bool,
        reports: dict,
        diffs: dict,
        baseline_mode: str,
    ) -> Path:
        results_dir = Path(conf.RETRIEVAL_EVAL_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = results_dir / f"retrieval_eval_{timestamp}.json"
        payload = {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": str(dataset),
            "modes": modes,
            "baseline_mode": baseline_mode,
            "top_k": top_k,
            "max_cases": max_cases,
            "use_hyde": use_hyde,
            "rerank": rerank,
            "reports": reports,
            "diffs": diffs,
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path


def build_default_tools() -> Dict[str, BaseTool]:
    """Create the fixed RAG MCP tool set."""
    tool_instances: Dict[str, BaseTool] = {
        tool.name: tool
        for tool in (
            EvidenceChunkRetrievalTool(),
            PaperOutlineTool(),
            PaperProfileTool(),
            PaperSummaryTool(),
            BuildPaperSummaryTool(),
            DbListTool(),
            DbSearchTool(),
            DbAddTool(),
            DbImportDirectoryTool(),
            ToolJobStatusTool(),
            RagHealthCheckTool(),
            DbDeleteTool(),
            DedupDatabaseTool(),
            BackfillMetadataTool(),
            RetrievalQualityEvalTool(),
        )
    }
    return tool_instances
