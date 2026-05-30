"""Multi-agent system for paper summarization, methodology analysis, code
reproduction, and literature search - implemented with LangGraph.

Agent dependencies are loaded lazily so web startup does not import LLM stacks
until chat execution actually needs them.
"""

__all__ = ["MultiAgentService"]


def __getattr__(name):
    if name == "MultiAgentService":
        from scholar_agent.agents.service import MultiAgentService

        return MultiAgentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
