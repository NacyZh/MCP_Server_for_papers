"""LangGraph graph construction for the modular multi-agent workflow."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scholar_agent.agents.executor import module_executor_node
from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.supervisor import supervisor_node
from scholar_agent.agents.synthesis import synthesis_node
from scholar_agent.core.logging import get_logger

logger = get_logger(__name__)


def _route_supervisor(state: MultiAgentState) -> str:
    """Route a supervisor plan to the module executor or END."""
    candidate = str(state.get("next_agent", "FINISH") or "").strip()
    if candidate.upper() == "FINISH":
        return "FINISH"
    candidate = candidate.lower()
    if candidate == "module_executor" or state.get("task_plan"):
        return "module_executor"
    logger.info("[agent] invalid supervisor route %r; forcing FINISH", candidate)
    return "FINISH"


def build_multi_agent_graph(max_steps: int = 12) -> StateGraph:
    """Build and compile the multi-agent graph.

    Flow:
        START -> supervisor(planner) -> module_executor -> synthesis -> END
                  |
                  +-> END  (direct answer / no module needed)

    Args:
        max_steps: Reserved for API compatibility with callers that configure
            recursion limits outside the compiled graph.
    """
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("module_executor", module_executor_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "module_executor": "module_executor",
            "FINISH": END,
        },
    )
    workflow.add_edge("module_executor", "synthesis")
    workflow.add_edge("synthesis", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


_graph: Any = None


def get_graph() -> Any:
    """Return the compiled multi-agent graph (lazy singleton)."""
    global _graph
    if _graph is None:
        _graph = build_multi_agent_graph()
    return _graph
