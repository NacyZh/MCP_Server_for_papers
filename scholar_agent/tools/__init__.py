"""MCP tool implementations for ScholarAgent.

Concrete tool modules are loaded lazily so importing a single lightweight tool
does not require optional retrieval, arXiv, or LangChain dependencies.
"""

__all__ = [
    "BaseTool",
    "ToolResult",
    "CodeWorkspaceApplyPatchTool",
    "CodeWorkspaceListTool",
    "CodeWorkspaceMakePatchTool",
    "CodeWorkspaceReadTool",
    "CodeWorkspaceRecordValidationTool",
    "CodeWorkspaceRunPytestTool",
    "CodeWorkspaceRunPythonTool",
    "CodeWorkspaceSetValidationPlanTool",
    "CodeWorkspaceWriteTool",
    "ArxivDownloadTool",
    "ArxivSearchTool",
    "BackfillMetadataTool",
    "DbAddTool",
    "DbDeleteTool",
    "DbListTool",
    "DbSearchTool",
    "DedupDatabaseTool",
    "LocalPaperChunksTool",
    "LocalSearchTool",
    "build_default_tools",
    "ToolRegistry",
    "get_tool_registry",
]


def __getattr__(name):
    if name in {"BaseTool", "ToolResult"}:
        from scholar_agent.tools.base import BaseTool, ToolResult

        return {"BaseTool": BaseTool, "ToolResult": ToolResult}[name]
    if name.startswith("CodeWorkspace"):
        from scholar_agent.tools import code_tools

        return getattr(code_tools, name)
    if name in {
        "ArxivDownloadTool",
        "ArxivSearchTool",
        "BackfillMetadataTool",
        "DbAddTool",
        "DbDeleteTool",
        "DbListTool",
        "DbSearchTool",
        "DedupDatabaseTool",
        "LocalPaperChunksTool",
        "LocalSearchTool",
        "build_default_tools",
    }:
        from scholar_agent.tools import paper_tools

        return getattr(paper_tools, name)
    if name in {"ToolRegistry", "get_tool_registry"}:
        from scholar_agent.tools.registry import ToolRegistry, get_tool_registry

        return {"ToolRegistry": ToolRegistry, "get_tool_registry": get_tool_registry}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
