"""FastAPI web server for ScholarAgent with multi-agent chat and direct tool endpoints."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scholar_agent.config import conf
from scholar_agent.core.logging import configure_logging, get_logger
from scholar_agent.tools.base import execute_tool_safely
from scholar_agent.tools.registry import get_tool_registry

if TYPE_CHECKING:
    from scholar_agent.agents.service import MultiAgentService

LOG_FILE = configure_logging()
logger = get_logger(__name__)
logger.info("web server logging initialized log_file=%s", LOG_FILE)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
logger.info("web server module loaded log_file=%s", LOG_FILE)


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=30000)


class ChatRequest(BaseModel):
    """Unified chat request for the multi-agent supervisor workflow."""
    message: str = Field(..., min_length=1, max_length=12000)
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    session_id: str | None = Field(default=None, max_length=120)
    code_workspace_path: str | None = Field(default=None, max_length=260)
    code_workspace_is_project: bool = False
    code_python_executable: str | None = Field(default=None, max_length=260)
    temperature: float = Field(default=0.3, ge=0.0, le=1.5)
    max_steps: int = Field(default=12, ge=1, le=40)


class DirectoryItem(BaseModel):
    name: str
    path: str


class FileItem(BaseModel):
    name: str
    path: str
    is_dir: bool = False


# ---------------------------------------------------------------------------
# Service singletons
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_service() -> "MultiAgentService":
    """Return the singleton multi-agent service instance."""
    from scholar_agent.agents.service import MultiAgentService

    return MultiAgentService()


def _execute_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
    """Execute a named tool from the registry and return a unified dict."""
    registry = get_tool_registry(discover_external=False)
    tool = registry.get(name)
    if tool is None:
        raise KeyError(f"Tool not found: {name}")
    result = execute_tool_safely(tool, kwargs)
    return {"tool": name, "input": kwargs, "status": result.status, "result": result.result}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="ScholarAgent Web API", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("shutdown")
def shutdown_external_resources() -> None:
    from scholar_agent.core.runtime import request_shutdown
    from scholar_agent.mcp_client import shutdown_external_tools

    request_shutdown("FastAPI shutdown")
    shutdown_external_tools()


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(path=str(STATIC_DIR / "favicon.svg"), media_type="image/svg+xml")


@app.get("/")
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Missing static/index.html")
    return FileResponse(path=str(INDEX_FILE))


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/filesystem/roots")
def filesystem_roots() -> Dict[str, Any]:
    """Return local filesystem roots for the workspace directory picker."""
    roots: list[DirectoryItem] = []
    if conf.IS_WINDOWS:
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            path = Path(f"{letter}:/")
            if path.exists():
                roots.append(DirectoryItem(name=f"{letter}:/", path=str(path)))
    else:
        roots.append(DirectoryItem(name="/", path="/"))
    home = Path.home()
    if home.exists():
        roots.append(DirectoryItem(name=f"Home ({home})", path=str(home)))
    work_root = Path(conf.SCHOLAR_AGENT_WORK_ROOT).expanduser()
    roots.append(DirectoryItem(name=f"ScholarAgent Work Root ({work_root})", path=str(work_root)))
    default_workspace = Path(conf.CODE_BUILDER_WORKSPACE_DIR).expanduser()
    roots.append(DirectoryItem(name=f"Code Workspace ({default_workspace})", path=str(default_workspace)))
    writing_workspace = Path(conf.WRITING_WORKSPACE_DIR).expanduser()
    roots.append(DirectoryItem(name=f"Document Workspace ({writing_workspace})", path=str(writing_workspace)))
    unique: dict[str, DirectoryItem] = {}
    for item in roots:
        unique[item.path] = item
    return {"roots": [item.model_dump() for item in unique.values()]}


@app.get("/api/filesystem/directories")
def filesystem_directories(path: str = Query(..., min_length=1, max_length=500)) -> Dict[str, Any]:
    """List child directories for a local path."""
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {resolved}")

    children: list[DirectoryItem] = []
    try:
        for child in sorted(resolved.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir():
                children.append(DirectoryItem(name=child.name, path=str(child)))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {resolved}") from exc

    parent = resolved.parent if resolved.parent != resolved else None
    return {
        "path": str(resolved),
        "parent": str(parent) if parent else "",
        "directories": [item.model_dump() for item in children],
    }


@app.get("/api/filesystem/python-files")
def filesystem_python_files(path: str = Query(..., min_length=1, max_length=500)) -> Dict[str, Any]:
    """List child directories and likely Python executables for the interpreter picker."""
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc
    selected_file = str(resolved) if resolved.is_file() else ""
    if selected_file:
        resolved = resolved.parent
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {resolved}")

    entries: list[FileItem] = []
    try:
        for child in sorted(resolved.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if child.is_dir():
                entries.append(FileItem(name=child.name, path=str(child), is_dir=True))
            elif child.name.lower() in {"python.exe", "pythonw.exe", "python"}:
                entries.append(FileItem(name=child.name, path=str(child), is_dir=False))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {resolved}") from exc

    parent = resolved.parent if resolved.parent != resolved else None
    return {
        "path": str(resolved),
        "selected_file": selected_file,
        "parent": str(parent) if parent else "",
        "entries": [item.model_dump() for item in entries],
    }


# ---------------------------------------------------------------------------
# Direct tool endpoints (no LLM — REST access to individual tools)
# ---------------------------------------------------------------------------


@app.get("/api/tools")
def list_tools() -> Dict[str, Any]:
    """List all available tool names."""
    registry = get_tool_registry(discover_external=False)
    return {"tools": registry.list_all()}


@app.get("/api/tools/schemas")
def list_tool_schemas() -> Dict[str, Any]:
    """List all available tools with their MCP-compatible schemas."""
    registry = get_tool_registry(discover_external=False)
    return {"tools": registry.list_schemas()}


@app.get("/api/papers/files")
def list_pdf_files() -> Dict[str, Any]:
    """List PDF files in the papers directory."""
    conf.check_config()
    paper_dir = Path(conf.PAPERS_DIR)
    if not paper_dir.exists():
        return {"files": []}
    files = sorted([item.name for item in paper_dir.glob("*.pdf")])
    return {"files": files}


# ---------------------------------------------------------------------------
# Dynamic tool endpoint generation
# ---------------------------------------------------------------------------

# Route map for tools that have established URL paths (backward-compatible).
# Tools not listed here get a generic /api/tools/{name} POST route.
_TOOL_ROUTE_MAP: Dict[str, tuple] = {
    "search_arxiv_papers":       ("POST", "/api/arxiv/search"),
    "search_local_papers_chunks": ("POST", "/api/local/search"),
    "search_local_database":     ("POST", "/api/local/database/search"),
    "list_local_database":       ("GET",  "/api/local/database"),
    "add_paper_to_database":     ("POST", "/api/papers/import"),
    "dedup_local_database":      ("POST", "/api/papers/dedup"),
    "backfill_paper_metadata":   ("POST", "/api/papers/backfill"),
    "download_arxiv_papers":     ("POST", "/api/arxiv/download"),
    "delete_paper_from_database": ("POST", "/api/papers/delete"),
}


def _make_tool_endpoint(tool_name: str):
    """Create a FastAPI endpoint handler for the given tool name.

    Each call creates its own scope so *tool_name* and *tool* are
    captured by closure.
    """
    registry = get_tool_registry(discover_external=False)
    tool = registry.get(tool_name)
    if tool is None:
        return None

    has_params = bool(tool.params.get("properties"))

    if has_params:
        def _handler(body: dict):
            result = execute_tool_safely(tool, body)
            return {"tool": tool_name, "input": body, "status": result.status, "result": result.result}
    else:
        def _handler():
            result = execute_tool_safely(tool, {})
            return {"tool": tool_name, "input": {}, "status": result.status, "result": result.result}

    _handler.__name__ = f"auto_{tool_name}"
    return _handler


def _generate_tool_endpoints(app: FastAPI):
    """Auto-register REST endpoints for all registered tools."""
    registry = get_tool_registry(discover_external=False)

    for name in registry.list_all():
        handler = _make_tool_endpoint(name)
        if handler is None:
            continue

        method, path = _TOOL_ROUTE_MAP.get(name, ("POST", f"/api/tools/{name}"))
        app.add_api_route(path, handler, methods=[method])


_generate_tool_endpoints(app)


# ---------------------------------------------------------------------------
# Chat endpoint (multi-agent supervisor workflow)
# ---------------------------------------------------------------------------


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    """Multi-agent chat — routes through the supervisor workflow.

    Flow: Supervisor → selected expert modules → Synthesis → Final answer.

    The supervisor has access to all tools (search, import, dedup, backfill,
    download) and can use them as needed during task planning.
    """
    service = get_service()
    history = [item.model_dump() for item in payload.history]
    try:
        result = service.chat(
            message=payload.message,
            history=history,
            session_id=payload.session_id,
            code_workspace_path=payload.code_workspace_path,
            code_workspace_is_project=payload.code_workspace_is_project,
            code_python_executable=payload.code_python_executable,
            temperature=payload.temperature,
            max_steps=payload.max_steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """Multi-agent chat with real-time SSE progress.

    Yields ``text/event-stream`` events as the supervisor routes through
    experts, so the frontend can show live progress instead of waiting
    for the entire workflow to finish.
    """
    service = get_service()
    history = [item.model_dump() for item in payload.history]

    def generate():
        yield from service.chat_stream(
            message=payload.message,
            history=history,
            session_id=payload.session_id,
            code_workspace_path=payload.code_workspace_path,
            code_workspace_is_project=payload.code_workspace_is_project,
            code_python_executable=payload.code_python_executable,
            temperature=payload.temperature,
            max_steps=payload.max_steps,
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/chat/memory/{session_id}")
def clear_chat_memory(session_id: str) -> Dict[str, Any]:
    """Clear persisted agent memory for one chat session."""
    service = get_service()
    return service.clear_memory(session_id)


@app.get("/api/chat/memory/{session_id}")
def get_chat_memory(session_id: str) -> Dict[str, Any]:
    """Inspect persisted agent memory for one chat session."""
    service = get_service()
    return service.get_memory(session_id)
