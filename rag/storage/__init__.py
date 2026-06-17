"""Storage layer: paper metadata storage and vector search.

Heavy vector dependencies are loaded only when their concrete modules are
imported directly.
"""

__all__ = [
    "PaperManager",
    "PaperDB",
    "VectorDB",
    "get_chunks_for_paper_ids_readonly",
    "list_papers_readonly",
]


def __getattr__(name):
    if name == "PaperManager":
        from rag.storage.paper_manager import PaperManager

        return PaperManager
    if name == "PaperDB":
        from rag.storage.sqlite_store import PaperDB

        return PaperDB
    if name == "VectorDB":
        from rag.storage.vector_store import VectorDB

        return VectorDB
    if name in {"get_chunks_for_paper_ids_readonly", "list_papers_readonly"}:
        from rag.storage import paper_manager

        return getattr(paper_manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
