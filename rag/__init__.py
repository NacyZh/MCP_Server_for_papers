__all__ = ["ScholarMCPServer", "PaperDB", "PaperManager", "VectorDB"]


def __getattr__(name):
    if name == "ScholarMCPServer":
        from rag.mcp_server import ScholarMCPServer

        return ScholarMCPServer
    if name in {"PaperDB", "PaperManager", "VectorDB"}:
        from rag import storage

        return getattr(storage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
