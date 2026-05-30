"""Concrete tool implementations for ScholarAgent.

Tools exposed:
- LocalSearchTool: Semantic + keyword search in local ChromaDB
- LocalPaperChunksTool: Read chunks for known local paper IDs
- ArxivSearchTool: Search arXiv API
- ArxivDownloadTool: Download PDF by arXiv ID
- DbListTool: List all papers in local database
- DbSearchTool: Search paper metadata records in local database
- DbAddTool: Parse and ingest a PDF (with duplicate policy)
- DbDeleteTool: Delete from SQLite + ChromaDB
- DedupDatabaseTool: Scan the database for duplicate papers
- BackfillMetadataTool: Recompute dedup columns for legacy rows
"""

import os
import random
import time
from typing import Dict, Literal, Optional, TYPE_CHECKING

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.storage.paper_manager import PaperManager
from scholar_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)

if TYPE_CHECKING:
    from scholar_agent.plugins.arxiv_search import ArxivManager
    from scholar_agent.storage.sqlite_store import PaperDB


class LocalSearchTool(BaseTool):
    name = "search_local_papers_chunks"
    description = (
        "Search the local paper database with semantic and keyword retrieval. "
        "Use this only with the user's academic topic, paper title, paper ID, or concise search keywords. "
        "Do not pass planning text, module instructions, or answer-format requirements as query."
    )
    params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Concise academic search query from the user's request, e.g. "
                    "'SCMA detection message passing deep learning receiver'. "
                    "Exclude workflow instructions such as summarize, analyze, generate code, cite IDs, or output language."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of relevant chunks to return (1-10, default 5).",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            logger.info("[tool] init local knowledge deps")
            self.pm = PaperManager()

    def execute(self, query: str, top_k: int = 3) -> ToolResult:
        self._ensure_pm()
        logger.info(f"[tool] search_local_papers query={query}")
        try:
            top_k = max(1, min(int(top_k), 10))
        except (TypeError, ValueError):
            top_k = 3
        try:
            results = self.pm.search_knowledge(query, n_results=top_k)
        except Exception as e:
            return ToolResult("fail", f"Local search failed: {e}", data=[])

        if not results:
            return ToolResult("fail", "No relevant content found in local database.", data=[])

        context = ""
        for i, res in enumerate(results):
            context += f"[Paper {i + 1}] (Title: {res['title']})\nContent: {res['content']}\n\n"
        return ToolResult("success", context, data=results)


class LocalPaperChunksTool(BaseTool):
    name = "get_local_paper_chunks"
    description = (
        "Read full-text chunks for known local paper IDs. "
        "Use this when a paper_id is available from user input, prior expert outputs, "
        "or conversation memory."
    )
    params = {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "string",
                "description": "Comma-separated local paper IDs, e.g. 'local_abc123,local_def456'.",
            },
            "max_chunks_per_paper": {
                "type": "integer",
                "description": "Maximum chunks to read per paper (1-20, default 8).",
                "minimum": 1,
                "maximum": 20,
                "default": 8,
            },
        },
        "required": ["paper_ids"],
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            logger.info("[tool] init local paper chunk deps")
            self.pm = PaperManager()

    def execute(self, paper_ids: str, max_chunks_per_paper: int = 8) -> ToolResult:
        self._ensure_pm()
        ids = [pid.strip() for pid in str(paper_ids or "").split(",") if pid.strip()]
        if not ids:
            return ToolResult("fail", "paper_ids must include at least one local paper ID", data=[])
        try:
            max_chunks_per_paper = max(1, min(int(max_chunks_per_paper), 20))
        except (TypeError, ValueError):
            max_chunks_per_paper = 8

        logger.info("[tool] get_local_paper_chunks paper_ids=%s", ids)
        try:
            results = self.pm.get_chunks_for_paper_ids(ids, max_chunks_per_paper=max_chunks_per_paper)
        except Exception as e:
            return ToolResult("fail", f"Reading local paper chunks failed: {e}", data=[])

        if not results:
            return ToolResult("fail", f"No chunks found for paper_ids: {', '.join(ids)}", data=[])

        context = ""
        for i, res in enumerate(results):
            context += f"[Chunk {i + 1}] (Paper ID: {res['paper_id']} | Title: {res['title']})\n"
            context += f"Content: {res['content']}\n\n"
        return ToolResult("success", context, data=results)


class ArxivSearchTool(BaseTool):
    name = "search_arxiv_papers"
    description = (
        "Search arXiv for academic preprints with sorting and result count control. "
        "The query must be concise English academic keywords or an arXiv advanced query. "
        "Do not pass Chinese task descriptions, planning text, module instructions, or answer-format requirements."
    )
    params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Concise English search keywords or arXiv advanced query, e.g. "
                    "'SCMA detection message passing deep learning NOMA receiver'. "
                    "Translate the research topic to English if the user asks in Chinese; "
                    "exclude instructions such as summarize, analyze, generate MATLAB/Python code, or cite papers."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Number of papers to return (1-5, default 3).",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
            "sort_by": {
                "type": "string",
                "description": "Sort by: relevance, submitted_date, or last_updated_date.",
                "enum": ["relevance", "submitted_date", "last_updated_date"],
                "default": "submitted_date",
            },
            "sort_order": {
                "type": "string",
                "description": "Sort order: descending (newest first) or ascending.",
                "enum": ["descending", "ascending"],
                "default": "descending",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, arxiv_searcher: Optional["ArxivManager"] = None):
        self.arxiv_searcher = arxiv_searcher

    def _ensure_arxiv_searcher(self):
        if self.arxiv_searcher is None:
            logger.info("[tool] init arxiv deps")
            from scholar_agent.plugins.arxiv_search import ArxivManager

            self.arxiv_searcher = ArxivManager()

    def execute(
        self,
        query: str,
        max_results: int = 3,
        sort_by: Literal["relevance", "submitted_date", "last_updated_date"] = "submitted_date",
        sort_order: Literal["descending", "ascending"] = "descending",
    ) -> ToolResult:
        self._ensure_arxiv_searcher()
        logger.info(
            f"[tool] search_arxiv_papers query={query} max_results={max_results} sort_by={sort_by} sort_order={sort_order}"
        )
        try:
            max_results = max(1, min(int(max_results), 10))
        except (TypeError, ValueError):
            max_results = 3
        sort_by = str(sort_by)
        sort_order = str(sort_order)
        results = self.arxiv_searcher.search_papers(
            query, max_results=max_results, sort_by=sort_by, sort_order=sort_order
        )
        if not results:
            detail = (self.arxiv_searcher.last_error or "").strip()
            if detail:
                return ToolResult("fail", detail, data=[])
            return ToolResult("fail", "No relevant papers found on arXiv.", data=[])
        return ToolResult("success", self.arxiv_searcher.format_for_llm(results), data=results)


class ArxivDownloadTool(BaseTool):
    name = "download_arxiv_papers"
    description = "Download a paper PDF from arXiv by its ID. Note: arXiv rate-limits; download serially."
    params = {
        "type": "object",
        "properties": {
            "arxiv_id": {"type": "string", "description": "arXiv paper ID (e.g., '2301.12345')."},
            "filename": {"type": "string", "description": "Filename without .pdf extension."},
            "dirpath": {"type": "string", "description": "Download directory (default: configured PAPERS_DIR)."},
        },
        "required": ["arxiv_id", "filename"],
        "additionalProperties": False,
    }

    def __init__(self, arxiv_searcher: Optional["ArxivManager"] = None):
        self.arxiv_searcher = arxiv_searcher

    def _ensure_arxiv_searcher(self):
        if self.arxiv_searcher is None:
            logger.info("[tool] init arxiv deps")
            from scholar_agent.plugins.arxiv_search import ArxivManager

            self.arxiv_searcher = ArxivManager()

    def execute(self, arxiv_id: str, filename: str, dirpath: str = conf.PAPERS_DIR) -> ToolResult:
        self._ensure_arxiv_searcher()
        status, msg = self.arxiv_searcher.download_paper_by_id(
            arxiv_id=arxiv_id, filename=filename, dirpath=dirpath
        )
        sleep_s = random.uniform(3, 8)
        time.sleep(sleep_s)
        return ToolResult(status, msg)


class DbListTool(BaseTool):
    name = "list_local_database"
    description = "List all papers currently stored in the local database."
    params = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, paper_manager: Optional[PaperManager] = None, paper_db: Optional["PaperDB"] = None):
        self.pm = paper_manager
        self.db = paper_db

    def _ensure_pm(self):
        if self.pm is None and self.db is None:
            logger.info("[tool] init sqlite db deps")
            from scholar_agent.storage.sqlite_store import PaperDB

            self.db = PaperDB()

    def execute(self) -> ToolResult:
        self._ensure_pm()
        logger.info("[tool] list_local_database")
        papers = self.pm.list_all() if self.pm is not None else self.db.get_all_papers()
        if not papers:
            return ToolResult("success", "The local database is empty.", data=[])
        res = "=== Local Database Papers ===\n"
        for p in papers:
            res += f"- ID: {p['id']} | Title: {p['title']}\n"
        return ToolResult("success", res, data=papers)


class DbSearchTool(BaseTool):
    name = "search_local_database"
    description = (
        "Search local paper metadata records by title, author, abstract, or tags. "
        "Use this to find paper IDs and metadata in the local SQLite paper library; "
        "use search_local_papers_chunks when full-text semantic chunks are needed."
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
            logger.info("[tool] init sqlite db deps")
            from scholar_agent.storage.sqlite_store import PaperDB

            self.db = PaperDB()

    def execute(self, query: str, limit: int = 20) -> ToolResult:
        self._ensure_db()
        query = str(query or "").strip()
        if not query:
            return ToolResult("fail", "query must not be empty", data=[])
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 20

        logger.info("[tool] search_local_database query=%s limit=%s", query, limit)
        papers = self.db.search_papers(query)[:limit]
        if not papers:
            return ToolResult("success", f"No local paper records matched query: {query}", data=[])

        lines = [f"=== Local Database Search Results ({len(papers)}) ==="]
        for p in papers:
            authors = (p.get("authors") or "").strip()
            year = p.get("publish_year") or "unknown year"
            tags = (p.get("tags") or "").strip()
            detail = f"- ID: {p['id']} | Title: {p['title']} | Year: {year}"
            if authors:
                detail += f" | Authors: {authors}"
            if tags:
                detail += f" | Tags: {tags}"
            lines.append(detail)
        return ToolResult("success", "\n".join(lines), data=papers)


class DbAddTool(BaseTool):
    name = "add_paper_to_database"
    description = (
        "Parse a PDF file and import it into the local database and vector store. "
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
        },
        "required": ["filename"],
        "additionalProperties": False,
    }

    def __init__(self, paper_manager: Optional[PaperManager] = None):
        self.pm = paper_manager

    def _ensure_pm(self):
        if self.pm is None:
            logger.info("[tool] init ingest deps")
            self.pm = PaperManager()

    def execute(self, filename: str, on_duplicate: str = "skip") -> ToolResult:
        self._ensure_pm()
        logger.info(f"[tool] add_paper_to_database filename={filename} on_duplicate={on_duplicate}")
        if not filename.lower().endswith(".pdf"):
            return ToolResult("fail", "filename must end with .pdf")
        if on_duplicate not in ("skip", "replace", "keep_both"):
            return ToolResult("fail", f"Invalid on_duplicate: {on_duplicate!r}")

        os.makedirs(conf.PAPERS_DIR, exist_ok=True)
        pdf_path = os.path.join(conf.PAPERS_DIR, filename)
        success, msg = self.pm.ingest_pdf(pdf_path, tags="Agent-Added", on_duplicate=on_duplicate)
        return ToolResult("success" if success else "fail", msg)


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
            logger.info("[tool] init ingest deps")
            self.pm = PaperManager()

    def execute(self, local_id: str) -> ToolResult:
        self._ensure_pm()
        logger.info(f"[tool] delete_paper local_id={local_id}")
        status, msg = self.pm.delete_paper(local_id)
        return ToolResult("success" if status else "fail", msg)


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
            logger.info("[tool] init dedup deps")
            self.pm = PaperManager()

    def execute(self, strategy: str = "report", backfill: bool = True) -> ToolResult:
        self._ensure_pm()
        logger.info(f"[tool] dedup_local_database strategy={strategy} backfill={backfill}")
        if strategy not in ("report", "prune"):
            return ToolResult("fail", f"Invalid strategy: {strategy!r}. Use 'report' or 'prune'.")
        try:
            report = self.pm.deduplicate_database(strategy=strategy, backfill=bool(backfill))
        except Exception as e:
            return ToolResult("fail", f"Dedup scan failed: {e}")

        groups = report["groups"]
        pruned = report["pruned"]
        total = report["total_papers"]

        if not groups:
            return ToolResult(
                "success",
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

        return ToolResult("success", "\n".join(lines))


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
            logger.info("[tool] init backfill deps")
            self.pm = PaperManager()

    def execute(self) -> ToolResult:
        self._ensure_pm()
        logger.info("[tool] backfill_paper_metadata")
        try:
            report = self.pm.backfill_metadata()
        except Exception as e:
            return ToolResult("fail", f"Backfill failed: {e}")

        scanned = report.get("scanned", 0)
        updated = report.get("updated", 0)
        missing = report.get("missing_files", 0)

        msg = (
            f"Backfill complete: scanned {scanned} paper(s), "
            f"updated {updated} with dedup metadata."
        )
        if missing:
            msg += f" {missing} paper(s) have missing files and could not be hashed."
        return ToolResult("success", msg)


def build_default_tools() -> Dict[str, BaseTool]:
    """Create tool instances with lazy dependency initialization.

    Auto-discovers all ``BaseTool`` subclasses registered via
    ``__init_subclass__``, including tools from external MCP servers.
    No hardcoded list — any new subclass is automatically included.
    """
    # Ensure external MCP tools are discovered before building
    import scholar_agent.tools.code_tools  # noqa: F401
    from scholar_agent.mcp_client import discover_external_tools
    discover_external_tools()

    tool_instances: Dict[str, BaseTool] = {}
    for name, cls in BaseTool._registry.items():
        tool_instances[name] = cls()
    return tool_instances
