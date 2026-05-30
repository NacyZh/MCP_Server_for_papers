"""Base classes and safe dispatch helpers for tool implementations."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Type


class ToolResult:
    """Unified result object for tool executions."""

    def __init__(self, status: str, result: str, data: Any = None):
        self.status = status  # "success" or "fail"
        self.result = result  # Human-readable / LLM-friendly text
        self.data = data      # Optional structured data (list of dicts, etc.)


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

    def execute(self, **kwargs) -> ToolResult:
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
    filtered, dropped, missing = normalize_tool_arguments(tool, kwargs)
    if missing:
        return ToolResult("fail", f"Missing required argument(s): {', '.join(missing)}")

    try:
        result = tool.execute(**filtered)
    except TypeError as exc:
        return ToolResult("fail", f"Parameter error: {exc}")
    except Exception as exc:
        return ToolResult("fail", f"Tool execution error: {exc}")

    if dropped:
        note = f"Ignored unexpected argument(s): {', '.join(dropped)}."
        return ToolResult(result.status, f"{result.result}\n{note}", data=result.data)
    return result
