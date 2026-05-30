"""Central tool registry with auto-discovery and LangChain bridge.

All ``BaseTool`` subclasses are automatically registered via
``__init_subclass__``.  This registry provides the single source of
truth for all three tool consumers: the MCP server, the supervisor
agent, and the web REST API.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from scholar_agent.tools.base import BaseTool, execute_tool_safely


def _json_schema_to_pydantic(
    params: dict,
    model_name: str,
) -> Type[BaseModel]:
    """Convert a JSON Schema ``params`` dict into a Pydantic model.

    Handles the subset of JSON Schema used by tool definitions:
    ``type``, ``properties``, ``required``, ``enum``, ``default``,
    and empty schemas for no-argument tools.
    """
    properties = params.get("properties", {})
    required: List[str] = params.get("required", [])

    fields: Dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        description = prop_schema.get("description", "")

        # Map JSON Schema type to Python type
        if json_type == "integer":
            py_type = int
        elif json_type == "number":
            py_type = float
        elif json_type == "boolean":
            py_type = bool
        else:
            py_type = str

        # Handle enum constraints via typing.Literal
        if "enum" in prop_schema:
            from typing import Literal
            py_type = Literal[tuple(prop_schema["enum"])]  # type: ignore[assignment]

        # Determine default value
        default_value = prop_schema.get("default")
        is_required = prop_name in required

        if is_required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            if default_value is not None:
                fields[prop_name] = (py_type, Field(default=default_value, description=description))
            elif json_type in ("integer", "number"):
                fields[prop_name] = (Optional[py_type], Field(default=None, description=description))  # type: ignore[call-overload]
            elif json_type == "boolean":
                fields[prop_name] = (Optional[py_type], Field(default=None, description=description))  # type: ignore[call-overload]
            else:
                fields[prop_name] = (Optional[py_type], Field(default=None, description=description))  # type: ignore[call-overload]

    if not fields:
        # No-arg tool: create a model with no fields
        return create_model(model_name, __base__=BaseModel)

    return create_model(model_name, **fields, __base__=BaseModel)


class ToolRegistry:
    """Singleton registry that auto-discovers ``BaseTool`` subclasses.

    Provides tool access, MCP-compatible schema listing, and
    LangChain ``StructuredTool`` generation for agent use.
    """

    def __init__(self, discover_external: bool = True):
        self._ensure_imported()
        if discover_external:
            self._discover_external()
        self._instances: Dict[str, BaseTool] = {}

    @staticmethod
    def _ensure_imported():
        """Trigger import of paper_tools so subclasses register themselves."""
        import scholar_agent.tools.code_tools  # noqa: F401
        import scholar_agent.tools.paper_tools  # noqa: F401
        import scholar_agent.tools.writing_tools  # noqa: F401

    @staticmethod
    def _discover_external():
        """Discover tools from external MCP servers (if enabled)."""
        from scholar_agent.mcp_client import discover_external_tools
        discover_external_tools()

    # ---- tool access -------------------------------------------------------

    def list_all(self) -> List[str]:
        """Return all registered tool names in sorted order."""
        return sorted(BaseTool._registry.keys())

    def get(self, name: str) -> Optional[BaseTool]:
        """Return a lazily-instantiated tool instance by name."""
        if name not in self._instances:
            cls = BaseTool._registry.get(name)
            if cls is None:
                return None
            self._instances[name] = cls()
        return self._instances[name]

    # ---- MCP schemas -------------------------------------------------------

    def list_schemas(self) -> List[dict]:
        """Return MCP-compatible tool schemas for all registered tools."""
        return [
            {"name": name, "description": cls.description, "inputSchema": cls.params}
            for name, cls in BaseTool._registry.items()
        ]

    # ---- LangChain bridge --------------------------------------------------

    def to_langchain_tool(self, name: str) -> StructuredTool:
        """Convert a registered ``BaseTool`` into a LangChain ``StructuredTool``.

        Dynamically builds a Pydantic args model from the tool's JSON Schema
        ``params`` so the LLM sees precise parameter types and descriptions.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not registered: {name}")

        args_schema = _json_schema_to_pydantic(tool.params, f"{name}_args")

        def _run(**kwargs: Any) -> str:
            result = execute_tool_safely(tool, kwargs)
            if result.status == "success":
                return result.result
            return f"[status={result.status}] {result.result}"

        return StructuredTool.from_function(
            func=_run,
            name=tool.name,
            description=tool.description,
            args_schema=args_schema,
        )

    def get_all_langchain_tools(self) -> List[StructuredTool]:
        """Return all registered tools as LangChain ``StructuredTool`` instances.

        Suitable for passing directly to ``llm.bind_tools()``.
        """
        return [self.to_langchain_tool(name) for name in self.list_all()]

    def get_langchain_tool_map(self) -> Dict[str, StructuredTool]:
        """Return ``{tool_name: StructuredTool}`` for tool-call dispatch."""
        return {name: self.to_langchain_tool(name) for name in self.list_all()}


@lru_cache(maxsize=2)
def get_tool_registry(discover_external: bool = True) -> ToolRegistry:
    """Return the singleton ``ToolRegistry`` instance."""
    return ToolRegistry(discover_external=discover_external)
