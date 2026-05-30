"""Plugin integrations: arXiv search, PDF parsing, and HyDE query expansion.

Plugin dependencies are loaded lazily. In particular, importing this package
must not load marker/PyTorch unless PDF parsing is actually requested.
"""

__all__ = ["ArxivManager", "HyDEExpander", "get_hyde_expander", "PaperParser"]


def __getattr__(name):
    if name == "ArxivManager":
        from scholar_agent.plugins.arxiv_search import ArxivManager

        return ArxivManager
    if name in {"HyDEExpander", "get_hyde_expander"}:
        from scholar_agent.plugins.hyde import HyDEExpander, get_hyde_expander

        return {"HyDEExpander": HyDEExpander, "get_hyde_expander": get_hyde_expander}[name]
    if name == "PaperParser":
        from scholar_agent.plugins.pdf_parser import PaperParser

        return PaperParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
