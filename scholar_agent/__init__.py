"""ScholarAgent - A local academic paper research assistant.

Provides LLM-powered paper search, vector-based semantic retrieval,
arXiv integration, and MCP tool exposure.
"""

from scholar_agent.config import Config, conf
from scholar_agent.core.logging import configure_logging, get_logger

__all__ = ["Config", "conf", "configure_logging", "get_logger"]
