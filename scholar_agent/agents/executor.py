"""Module executor for the modular multi-agent workflow."""

from __future__ import annotations

from typing import Callable, Dict

from scholar_agent.agents.experts.code_builder import code_builder_node
from scholar_agent.agents.experts.database_manager import database_manager_node
from scholar_agent.agents.experts.literature import literature_node
from scholar_agent.agents.experts.methodology import methodology_node
from scholar_agent.agents.experts.summarizer import summarizer_node
from scholar_agent.agents.experts.writing_editor import writing_editor_node
from scholar_agent.agents.planning import default_task_for_agent, normalize_task_plan
from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.utils import extract_user_query
from scholar_agent.core.logging import get_logger

logger = get_logger(__name__)

_EXPERT_NODES: Dict[str, Callable[[MultiAgentState], dict]] = {
    "literature": literature_node,
    "summarizer": summarizer_node,
    "methodology": methodology_node,
    "code_builder": code_builder_node,
    "database_manager": database_manager_node,
    "writing_editor": writing_editor_node,
}


def module_executor_node(state: MultiAgentState) -> dict:
    """Execute the supervisor-selected expert modules without supervisor loops."""
    task_plan = normalize_task_plan(state.get("task_plan", []))
    module_tasks = state.get("module_tasks", {}) or {}
    if not task_plan:
        logger.info("[agent] module_executor skipped empty plan")
        return {"next_agent": "synthesis", "expert_outputs": []}

    user_query = extract_user_query(state).strip() or state.get("current_task", "")
    logger.info("[agent] module_executor start modules=%s", task_plan)

    new_outputs = []
    new_messages = []
    completed_outputs = list(state.get("expert_outputs", []))
    for agent in task_plan:
        node_fn = _EXPERT_NODES.get(agent)
        if node_fn is None:
            continue
        if _should_skip_for_failed_prerequisite(agent, completed_outputs + new_outputs, task_plan):
            logger.info("[agent] module_executor skip %s due to failed prerequisite", agent)
            emit_progress(
                "expert_skipped",
                agent=agent,
                reason="前置模块没有找到足够论文证据，已跳过该模块。",
            )
            continue
        task = str(module_tasks.get(agent, "")).strip() or default_task_for_agent(agent, user_query)
        local_state = dict(state)
        local_state["current_task"] = task
        local_state["expert_outputs"] = completed_outputs + new_outputs
        logger.info("[agent] module_executor -> %s", agent)
        emit_progress("expert_start", agent=agent, task=task)
        update = node_fn(local_state) or {}
        outputs = update.get("expert_outputs", []) or []
        messages = update.get("messages", []) or []
        new_outputs.extend(outputs)
        new_messages.extend(messages)
        emit_progress("expert_done", agent=agent, output_count=len(outputs))

    logger.info("[agent] module_executor done outputs=%s", len(new_outputs))
    return {
        "next_agent": "synthesis",
        "expert_outputs": new_outputs,
        "messages": new_messages,
    }


def _should_skip_for_failed_prerequisite(agent: str, expert_outputs: list[dict], task_plan: list[str]) -> bool:
    """Avoid running downstream modules when required paper evidence is absent."""
    if agent not in {"summarizer", "methodology", "code_builder"}:
        return False
    for output in expert_outputs:
        metadata = output.get("metadata", {}) or {}
        if metadata.get("error") == "no_papers_found":
            return True
    if agent == "code_builder":
        has_methodology = any(output.get("expert_name") == "methodology" for output in expert_outputs)
        if "methodology" in task_plan and not has_methodology:
            return True
    return False
