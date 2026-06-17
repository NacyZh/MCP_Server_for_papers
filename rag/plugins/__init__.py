"""Plugin integrations: PDF parsing and HyDE query expansion.

Plugin dependencies are loaded lazily. In particular, importing this package
must not load marker/PyTorch unless PDF parsing is actually requested.
"""

__all__ = ["HyDEExpander", "SummaryModelManager", "get_hyde_expander", "get_summary_model_manager", "PaperParser"]


def __getattr__(name):
    if name in {"HyDEExpander", "get_hyde_expander"}:
        from rag.plugins.hyde import HyDEExpander, get_hyde_expander

        return {"HyDEExpander": HyDEExpander, "get_hyde_expander": get_hyde_expander}[name]
    if name in {"SummaryModelManager", "get_summary_model_manager"}:
        from rag.plugins.summary_model import SummaryModelManager, get_summary_model_manager

        return {
            "SummaryModelManager": SummaryModelManager,
            "get_summary_model_manager": get_summary_model_manager,
        }[name]
    if name == "PaperParser":
        from rag.plugins.pdf_parser import PaperParser

        return PaperParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
