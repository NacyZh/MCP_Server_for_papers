"""Planning helpers for the modular multi-agent workflow."""

from __future__ import annotations

from typing import Any, Dict, List

from scholar_agent.config import conf


def normalize_agent_name(value: Any) -> str:
    """Return a canonical expert name or FINISH; return empty string if invalid."""
    candidate = str(value or "").strip()
    if candidate.upper() == "FINISH":
        return "FINISH"
    candidate = candidate.lower()
    if candidate in conf.EXPERT_NAMES:
        return candidate
    return ""


def normalize_task_plan(values: List[str]) -> List[str]:
    """Keep only valid expert names and remove duplicates while preserving order."""
    plan: List[str] = []
    for value in values or []:
        agent = normalize_agent_name(value)
        if agent and agent != "FINISH" and agent not in plan:
            plan.append(agent)
    return plan


def infer_task_plan(user_query: str) -> List[str]:
    """Return no inferred modules when the LLM planner is unavailable.

    Module selection is intentionally delegated to the supervisor LLM. This
    fallback keeps the graph safe without attempting keyword-based intent
    detection.
    """
    return []


def default_task_for_agent(agent: str, user_query: str) -> str:
    """Build a stable per-expert task when the LLM omits or misroutes one."""
    query = user_query or "用户未提供明确问题"
    templates = {
        "literature": "围绕用户问题检索本地数据库和 arXiv，筛选最相关论文: {query}",
        "summarizer": "基于相关论文内容，提炼与用户问题直接相关的背景、贡献、实验和结论: {query}",
        "methodology": "从相关论文中抽取方法细节、数学公式、算法流程和实验设置: {query}",
        "code_builder": "根据方法分析结果生成可运行的复现代码，并标注必要假设: {query}",
        "database_manager": "根据用户请求管理本地论文数据库，并基于工具执行结果回复: {query}",
        "writing_editor": "根据用户请求完成论文写作、翻译、结构化改写或中英文风格润色: {query}",
    }
    return templates.get(agent, "{query}").format(query=query)


def default_plan_reason(task_plan: List[str]) -> str:
    """Describe why a module plan is selected."""
    labels = {
        "literature": "文献检索",
        "summarizer": "论文总结",
        "methodology": "方法分析",
        "code_builder": "代码生成",
        "database_manager": "数据库管理",
        "writing_editor": "论文写作润色",
    }
    readable = " -> ".join(labels.get(agent, agent) for agent in task_plan)
    return f"统一调度模块执行计划: {readable}"


def normalize_module_tasks(raw_tasks: Any, task_plan: List[str], user_query: str) -> Dict[str, str]:
    """Validate module task instructions from the planner."""
    if not isinstance(raw_tasks, dict):
        raw_tasks = {}
    module_tasks: Dict[str, str] = {}
    for agent in task_plan:
        task = str(raw_tasks.get(agent, "")).strip()
        module_tasks[agent] = task or default_task_for_agent(agent, user_query)
    return module_tasks
