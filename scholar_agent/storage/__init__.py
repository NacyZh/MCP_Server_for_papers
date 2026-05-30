"""Storage layer: paper metadata storage and vector search.

Heavy vector dependencies are loaded only when their concrete modules are
imported directly.
"""

__all__ = ["PaperManager", "PaperDB", "VectorDB", "AgentMemoryStore", "get_agent_memory_store"]


def __getattr__(name):
    if name == "PaperManager":
        from scholar_agent.storage.paper_manager import PaperManager

        return PaperManager
    if name == "PaperDB":
        from scholar_agent.storage.sqlite_store import PaperDB

        return PaperDB
    if name == "VectorDB":
        from scholar_agent.storage.vector_store import VectorDB

        return VectorDB
    if name in {"AgentMemoryStore", "get_agent_memory_store"}:
        from scholar_agent.storage.memory_store import AgentMemoryStore, get_agent_memory_store

        return {"AgentMemoryStore": AgentMemoryStore, "get_agent_memory_store": get_agent_memory_store}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
