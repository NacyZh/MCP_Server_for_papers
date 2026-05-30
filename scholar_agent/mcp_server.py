"""MCP (Model Context Protocol) server exposing ScholarAgent tools to AI clients."""

import inspect
import sys
from pathlib import Path
from typing import Dict, Literal

if __package__ is None:
    _proj = Path(__file__).resolve().parent
    while _proj and not (_proj / "pyproject.toml").exists():
        _proj = _proj.parent
    if _proj and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

from mcp.server.fastmcp import FastMCP

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.tools.base import BaseTool, execute_tool_safely
from scholar_agent.tools.paper_tools import build_default_tools

logger = get_logger(__name__)


class ScholarMCPServer:
    """Standard MCP server wrapper for ScholarAgent tools."""

    def __init__(self):
        conf.check_config()
        self.tools: Dict[str, BaseTool] = build_default_tools()
        self.tool_names: list[str] = []
        self.app = FastMCP(name="ScholarAgent")
        self._register_tools()

    def get_registered_tool_names(self) -> list[str]:
        return list(self.tool_names)

    def _ensure_tools(self):
        if not self.tools:
            logger.info("[mcp] loading tool instances lazily")
            self.tools = build_default_tools()
            logger.info(f"[mcp] tools ready count={len(self.tools)}")

    def _build_tool_runner(self, tool_name: str, tool: BaseTool):
        execute_sig = inspect.signature(tool.execute)
        exposed_params = []
        for param in execute_sig.parameters.values():
            if param.name == "self":
                continue
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                exposed_params.append(param)

        runner_sig = inspect.Signature(parameters=exposed_params, return_annotation=str)

        def runner(**kwargs):
            return self._execute_tool(tool_name, **kwargs)

        runner.__name__ = tool_name
        runner.__doc__ = tool.description
        runner.__signature__ = runner_sig
        runner.__annotations__ = {p.name: p.annotation for p in exposed_params}
        runner.__annotations__["return"] = str
        return runner

    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        self._ensure_tools()
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"[status=fail] Tool {tool_name} not found."

        result = execute_tool_safely(tool, kwargs)
        if result.status == "success":
            return result.result
        return f"[status={result.status}] {result.result}"

    def _register_tools(self):
        self._ensure_tools()
        for tool_name, tool in self.tools.items():
            runner = self._build_tool_runner(tool_name, tool)
            self.app.add_tool(runner, name=tool_name, description=tool.description)
            self.tool_names.append(tool_name)

    def run(self, transport: Literal["stdio", "sse", "streamable-http"]):
        self.app.run(transport=transport)


def run_stdio_server():
    ScholarMCPServer().run(transport="stdio")


if __name__ == "__main__":
    run_stdio_server()
