"""MCP client — connects to external MCP servers and wraps their tools.

External MCP tools are dynamically wrapped as ``RemoteMCPTool`` subclasses,
which auto-register into ``BaseTool._registry`` via ``__init_subclass__``.
"""

from __future__ import annotations

import atexit
import asyncio
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%")


def _expand_env_value(value: Any) -> Any:
    """Expand environment variables in MCP config values."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2) or match.group(3) or ""
            return os.environ.get(name, match.group(0))

        return os.path.expanduser(_ENV_VAR_RE.sub(replace, value))
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    return value


# ============================================================================
# MCP Connection — one per external server
# ============================================================================


class MCPConnection:
    """A persistent stdio connection to one MCP server process."""

    def __init__(self, name: str, config: dict):
        self.name = name
        config = _expand_env_value(config)
        self.command = config.get("command", "")
        self.args = config.get("args", [])
        self.cwd = config.get("cwd") or os.getcwd()
        self._env = config.get("env") or {}
        self._timeout = config.get("timeout", 30)
        # Accept both "type" (Claude Code / VS Code style) and "transport"
        self.transport = config.get("transport") or config.get("type", "stdio")
        self._process: Optional[subprocess.Popen] = None
        self._tools: List[dict] = []

    # ---- lifecycle ---------------------------------------------------------

    def start(self):
        """Launch the server subprocess and perform the MCP handshake."""
        env = os.environ.copy()
        env.update(self._env)

        try:
            self._process = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=env,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError:
            logger.info(f"[mcp-client] command not found for '{self.name}': {self.command}")
            return
        except Exception as exc:
            logger.info(f"[mcp-client] failed to start '{self.name}': {exc}")
            return

        # MCP initialization handshake
        try:
            self._send_json({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ScholarAgent", "version": "0.3.0"},
                },
            })
            init_resp = self._recv_json()
            if "error" in init_resp:
                logger.info(f"[mcp-client] initialize failed for '{self.name}': {init_resp['error']}")
                return

            # Send initialized notification
            self._send_json({"jsonrpc": "2.0", "method": "notifications/initialized"})

            # Discover tools
            self._send_json({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            tools_resp = self._recv_json()
            self._tools = tools_resp.get("result", {}).get("tools", [])
            logger.info(
                f"[mcp-client] connected '{self.name}' — "
                f"{len(self._tools)} tool(s) discovered"
            )

        except Exception as exc:
            logger.info(f"[mcp-client] handshake failed for '{self.name}': {exc}")

    def stop(self):
        """Terminate the server subprocess."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.stdout.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

    # ---- tool operations ---------------------------------------------------

    def list_tools(self) -> List[dict]:
        """Return the list of tool schemas discovered from this server."""
        return list(self._tools)

    def call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Invoke a tool on the remote server and return a ToolResult."""
        if self._process is None or self._process.poll() is not None:
            return ToolResult("fail", f"MCP server '{self.name}' is not running")

        try:
            self._send_json({
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })
            resp = self._recv_json()
        except Exception as exc:
            return ToolResult("fail", f"MCP call '{tool_name}' on '{self.name}' failed: {exc}")

        if "error" in resp:
            return ToolResult("fail", f"MCP error: {resp['error']}")

        result = resp.get("result", {})
        content = result.get("content", [])
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return ToolResult("success", "\n".join(text_parts) or "(no output)")

    # ---- JSON-RPC wire helpers ---------------------------------------------

    def _send_json(self, data: dict):
        if self._process and self._process.stdin:
            raw = json.dumps(data, ensure_ascii=False)
            self._process.stdin.write(raw + "\n")
            self._process.stdin.flush()

    def _recv_json(self) -> dict:
        if self._process and self._process.stdout:
            line = self._process.stdout.readline()
            if not line:
                raise ConnectionError("MCP server closed stdout")
            return json.loads(line.strip())
        raise ConnectionError("MCP server not connected")


# ============================================================================
# RemoteMCPTool — wraps an external MCP tool as a BaseTool subclass
# ============================================================================


class RemoteMCPTool(BaseTool):
    """Base class for tools discovered from external MCP servers.

    Dynamic subclasses are created for each discovered tool; these
    subclasses carry the tool's ``name``, ``description``, and
    ``params`` (from the server's ``inputSchema``) and delegate
    ``execute()`` to the shared ``MCPConnection``.

    The base class has an empty *name* so it is **not** registered
    in the tool registry — only the dynamically-created per-tool
    subclasses carry non-empty names and get registered.
    """

    name = ""  # empty → skipped by BaseTool.__init_subclass__
    description: str = ""
    params: dict = {}

    # Shared connection pool — {server_name: MCPConnection}
    _connections: Dict[str, MCPConnection] = {}

    # Instance-level reference to the server
    _server_name: str = ""
    _tool_name_on_server: str = ""

    @classmethod
    def set_connection(cls, server_name: str, conn: MCPConnection):
        """Register a shared connection for all remote tools on this server."""
        cls._connections[server_name] = conn

    def execute(self, **kwargs) -> ToolResult:
        conn = self._connections.get(self._server_name)
        if conn is None:
            return ToolResult("fail", f"MCP server '{self._server_name}' not connected")
        return conn.call_tool(self._tool_name_on_server, kwargs)


# ============================================================================
# MCPClientManager — discovers and registers external MCP tools
# ============================================================================


class MCPClientManager:
    """Manages connections to all configured external MCP servers.

    On ``discover_all()``, connects to each server, fetches its tool
    list, and dynamically creates ``RemoteMCPTool`` subclasses that
    auto-register into ``BaseTool._registry``.
    """

    def __init__(self):
        self._connections: Dict[str, MCPConnection] = {}
        self._discovered = False

    def discover_all(self):
        """Connect to all configured servers and register their tools."""
        if self._discovered:
            return
        self._discovered = True

        servers = self._load_config()
        if not servers:
            logger.info("[mcp-client] no external MCP servers configured")
            return

        for server_name, server_config in servers.items():
            self._connect_server(server_name, server_config)

    def shutdown(self):
        """Stop all MCP server subprocesses."""
        for conn in self._connections.values():
            conn.stop()
        self._connections.clear()

    # ---- internal ----------------------------------------------------------

    def _load_config(self) -> dict:
        """Load MCP server configuration from a YAML or JSON config file.

        Tries the path in ``conf.MCP_SERVERS_CONFIG`` first, then the
        alternate format (``.yaml`` ↔ ``.json``).  If both exist, both
        are loaded and their ``servers`` are merged (JSON takes priority).
        """
        results: List[dict] = []
        primary = conf.MCP_SERVERS_CONFIG or ""

        # Collect candidate paths
        candidates = []
        if primary and os.path.exists(primary):
            candidates.append(primary)
        # Try alternate format
        if primary.endswith((".yaml", ".yml")):
            alt = primary[: primary.rindex(".")] + ".json"
        elif primary.endswith(".json"):
            alt = primary[: primary.rindex(".")] + ".yaml"
        else:
            alt = ""
        if alt and os.path.exists(alt) and alt not in candidates:
            candidates.append(alt)

        if not candidates:
            return {}

        for path in candidates:
            try:
                if path.endswith((".yaml", ".yml")):
                    import yaml
                    with open(path, "r", encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                elif path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                else:
                    continue
                if data.get("servers"):
                    results.append(data["servers"])
            except Exception as exc:
                logger.info(f"[mcp-client] failed to parse {path}: {exc}")

        # Merge: later files override earlier ones by server name
        merged: dict = {}
        for servers in results:
            merged.update(servers)
        return merged

    def _connect_server(self, name: str, config: dict):
        conn = MCPConnection(name, config)
        conn.start()
        if conn._process is None or conn._process.poll() is not None:
            logger.info(f"[mcp-client] server '{name}' failed to start, skipping")
            return

        self._connections[name] = conn
        RemoteMCPTool.set_connection(name, conn)

        for tool_schema in conn.list_tools():
            self._register_tool(name, conn, tool_schema)

    def _register_tool(self, server_name: str, conn: MCPConnection, schema: dict):
        """Dynamically create a RemoteMCPTool subclass for a discovered tool."""
        tool_name = schema.get("name", "")
        if not tool_name:
            return

        # Create a unique class name and a unique tool name (prefixed by server)
        cls_name = f"RemoteMCP_{server_name}_{tool_name}"
        # Sanitize: OpenAI function names only allow [a-zA-Z0-9_-]
        safe_server = server_name.replace(".", "_").replace("-", "_")
        safe_tool = tool_name.replace(".", "_").replace("-", "_")
        qualified_name = f"{safe_server}__{safe_tool}"

        subclass = type(
            cls_name,
            (RemoteMCPTool,),
            {
                "name": qualified_name,
                "description": schema.get("description", ""),
                "params": schema.get("inputSchema", {}),
                "is_external_mcp_tool": True,
                "_server_name": server_name,
                "_tool_name_on_server": tool_name,
            },
        )
        logger.info(f"[mcp-client] registered {qualified_name}")
        return subclass


# ============================================================================
# Module-level singleton
# ============================================================================

_manager: Optional[MCPClientManager] = None


def get_mcp_client_manager() -> MCPClientManager:
    """Return the singleton MCPClientManager, creating it if needed."""
    global _manager
    if _manager is None:
        _manager = MCPClientManager()
    return _manager


def discover_external_tools():
    """Discover and register tools from all configured external MCP servers."""
    if not conf.ENABLE_EXTERNAL_MCP:
        logger.info("[mcp-client] external MCP tools disabled (ENABLE_EXTERNAL_MCP=False)")
        return
    mgr = get_mcp_client_manager()
    mgr.discover_all()


def shutdown_external_tools(log: bool = True):
    """Stop all external MCP subprocesses if they were started."""
    global _manager
    from scholar_agent.core.runtime import request_shutdown

    if log:
        request_shutdown("external MCP shutdown")
    if _manager is None:
        return
    try:
        _manager.shutdown()
        if log:
            logger.info("[mcp-client] external MCP tools shut down")
    except Exception as exc:
        if log:
            logger.info("[mcp-client] external MCP shutdown failed: %s", exc)


atexit.register(lambda: shutdown_external_tools(log=False))
