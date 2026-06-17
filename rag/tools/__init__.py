"""MCP tool implementations for ScholarAgent.

Concrete tool modules are loaded lazily so importing a lightweight tool does
not require vector retrieval or PDF parsing dependencies.
"""

__all__ = [
    "BaseTool",
    "ToolResult",
    "BuildPaperSummaryTool",
    "DbAddTool",
    "BackfillMetadataTool",
    "DbDeleteTool",
    "DbListTool",
    "DbSearchTool",
    "DedupDatabaseTool",
    "EvidenceChunkRetrievalTool",
    "PaperOutlineTool",
    "PaperProfileTool",
    "PaperSummaryTool",
    "RagHealthCheckTool",
    "ToolJobStatusTool",
    "build_default_tools",
]


def __getattr__(name):
    if name in {"BaseTool", "ToolResult"}:
        from rag.tools.base import BaseTool, ToolResult

        return {"BaseTool": BaseTool, "ToolResult": ToolResult}[name]
    if name in {
        "BackfillMetadataTool",
        "BuildPaperSummaryTool",
        "DbAddTool",
        "DbDeleteTool",
        "DbListTool",
        "DbSearchTool",
        "DedupDatabaseTool",
        "EvidenceChunkRetrievalTool",
        "PaperOutlineTool",
        "PaperProfileTool",
        "PaperSummaryTool",
        "RagHealthCheckTool",
        "ToolJobStatusTool",
        "build_default_tools",
    }:
        from rag.tools import paper_tools

        return getattr(paper_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
