"""Base classes and safe dispatch helpers for tool implementations."""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Dict, Type

from rag.core.observability import log_event, new_request_id


class ToolResult:
    """Unified result object for tool executions."""

    def __init__(
        self,
        status: str,
        result: str,
        data: Any = None,
        error_code: str = "",
        recoverable: bool = True,
        suggestion: str = "",
        request_id: str = "",
        elapsed_ms: int | None = None,
    ):
        self.status = status  # "success" or "fail"
        self.result = result  # Human-readable / LLM-friendly text
        self.data = data      # Optional structured data (list of dicts, etc.)
        self.error_code = error_code
        self.recoverable = recoverable
        self.suggestion = suggestion
        self.request_id = request_id
        self.elapsed_ms = elapsed_ms

    @classmethod
    def success(
        cls,
        result: str,
        data: Any = None,
        request_id: str = "",
        elapsed_ms: int | None = None,
    ) -> "ToolResult":
        return cls("success", result, data=data, request_id=request_id, elapsed_ms=elapsed_ms)

    @classmethod
    def fail(
        cls,
        result: str,
        *,
        error_code: str,
        recoverable: bool = True,
        suggestion: str = "",
        data: Any = None,
        request_id: str = "",
        elapsed_ms: int | None = None,
    ) -> "ToolResult":
        return cls(
            "fail",
            result,
            data=data,
            error_code=error_code,
            recoverable=recoverable,
            suggestion=suggestion,
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )

    def to_payload(self) -> dict:
        payload = {
            "status": self.status,
            "result": self.result,
            "data": self.data,
        }
        if self.request_id:
            payload["request_id"] = self.request_id
        if self.elapsed_ms is not None:
            payload["elapsed_ms"] = self.elapsed_ms
        if self.status != "success":
            payload["error_code"] = self.error_code or "TOOL_FAILED"
            payload["recoverable"] = bool(self.recoverable)
            payload["suggestion"] = self.suggestion
        return payload

    def to_mcp_text(self) -> str:
        if self.status == "success":
            return self.result
        return "[tool_error] " + json.dumps(self.to_payload(), ensure_ascii=False)


class BaseTool:
    """Abstract base class for all concrete tools.

    Subclasses are automatically registered via ``__init_subclass__``
    into ``BaseTool._registry`` keyed by their ``name`` class attribute.
    """

    name: str = "base_tool"
    description: str = "Tool description."
    params: dict = {}  # JSON Schema for tool parameters

    _registry: Dict[str, Type["BaseTool"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", "")
        if name and name != "base_tool":
            BaseTool._registry[name] = cls

    def get_mcp_schema(self) -> dict:
        """Return MCP-compatible tool schema (name, description, inputSchema)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.params,
        }

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool logic. Subclasses must implement this."""
        raise NotImplementedError


def normalize_tool_arguments(tool: BaseTool, kwargs: dict | None) -> tuple[dict, list[str], list[str]]:
    """Filter LLM/tool endpoint arguments to the tool's declared input surface.

    LLMs can occasionally include stale or neighboring fields from another tool
    call. Tool schemas are the contract, so unknown arguments are ignored while
    missing required arguments are reported as a normal tool failure by
    :func:`execute_tool_safely` instead of bubbling up as a Python ``TypeError``.
    """
    raw = dict(kwargs or {}) if isinstance(kwargs, dict) else {}
    params = getattr(tool, "params", {}) or {}
    properties = params.get("properties")

    if isinstance(properties, dict):
        allowed = set(properties)
        filtered = {key: value for key, value in raw.items() if key in allowed}
        dropped = [key for key in raw if key not in allowed]
        required = [key for key in params.get("required", []) if key not in filtered]
        return filtered, dropped, required

    try:
        signature = inspect.signature(tool.execute)
    except (TypeError, ValueError):
        return raw, [], []

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return raw, [], []

    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {key: value for key, value in raw.items() if key in allowed}
    dropped = [key for key in raw if key not in allowed]
    required = [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and name not in filtered
        and param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    return filtered, dropped, required


def execute_tool_safely(tool: BaseTool, kwargs: dict | None = None) -> ToolResult:
    """Execute a tool with schema-based argument normalization."""
    request_id = new_request_id("tool")
    tool_name = getattr(tool, "name", tool.__class__.__name__)
    started = time.perf_counter()
    filtered, dropped, missing = normalize_tool_arguments(tool, kwargs)
    if missing:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            "tool_failed",
            request_id=request_id,
            tool=tool_name,
            error_code="MISSING_REQUIRED_ARGUMENT",
            elapsed_ms=elapsed_ms,
        )
        return ToolResult.fail(
            f"Missing required argument(s): {', '.join(missing)}",
            error_code="MISSING_REQUIRED_ARGUMENT",
            suggestion="Call the tool again with all required arguments from the schema.",
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )

    try:
        log_event("tool_started", request_id=request_id, tool=tool_name)
        result = tool.execute(**filtered)
    except TypeError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            "tool_failed",
            request_id=request_id,
            tool=tool_name,
            error_code="PARAMETER_ERROR",
            elapsed_ms=elapsed_ms,
        )
        return ToolResult.fail(
            f"Parameter error: {exc}",
            error_code="PARAMETER_ERROR",
            suggestion="Check the tool schema and retry with supported argument names and types.",
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            "tool_failed",
            request_id=request_id,
            tool=tool_name,
            error_code="UNHANDLED_TOOL_EXCEPTION",
            elapsed_ms=elapsed_ms,
        )
        return ToolResult.fail(
            f"Tool execution error: {exc}",
            error_code="UNHANDLED_TOOL_EXCEPTION",
            recoverable=False,
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result.request_id = result.request_id or request_id
    result.elapsed_ms = elapsed_ms
    log_event(
        "tool_finished",
        request_id=request_id,
        tool=tool_name,
        status=result.status,
        error_code=result.error_code,
        elapsed_ms=elapsed_ms,
    )

    if dropped:
        note = f"Ignored unexpected argument(s): {', '.join(dropped)}."
        return ToolResult(
            result.status,
            f"{result.result}\n{note}",
            data=result.data,
            error_code=result.error_code,
            recoverable=result.recoverable,
            suggestion=result.suggestion,
            request_id=result.request_id,
            elapsed_ms=result.elapsed_ms,
        )
    return result
