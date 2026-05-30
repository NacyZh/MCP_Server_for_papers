"""Supervisor planner for the modular multi-agent workflow."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.planning import (
    default_plan_reason,
    default_task_for_agent,
    infer_task_plan,
    normalize_agent_name,
    normalize_module_tasks,
    normalize_task_plan,
)
from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.utils import extract_user_query
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.prompts import SUPERVISOR_SYSTEM_PROMPT
from scholar_agent.skills.loader import get_skill_loader
from scholar_agent.storage.memory_store import AgentMemoryStore

logger = get_logger(__name__)


def supervisor_node(state: MultiAgentState) -> Dict[str, Any]:
    """Plan once, then hand execution to the module executor or answer directly."""
    logger.info("[agent] supervisor planning")

    user_query = extract_user_query(state).strip() or state.get("current_task", "")
    memory = state.get("memory", {})
    memory_text = AgentMemoryStore.format_for_prompt(memory)
    active_skills: List[str] = list(state.get("active_skills", []))
    if not active_skills and isinstance(memory, dict):
        active_skills = list(memory.get("active_skills", []))

    skill_overrides, skill_plan = _load_active_skill_overrides(active_skills)
    planning_query = f"{user_query}\n{memory_text}" if memory_text else user_query
    fallback_plan = normalize_task_plan(state.get("task_plan", []))
    if not fallback_plan:
        fallback_plan = skill_plan or infer_task_plan(planning_query)

    llm_plan = _plan_with_llm(
        user_query=user_query,
        memory_text=memory_text,
        fallback_plan=fallback_plan,
        active_skills=active_skills,
    )

    task_plan = fallback_plan
    module_tasks = {agent: default_task_for_agent(agent, user_query) for agent in task_plan}
    reason = default_plan_reason(task_plan)

    if llm_plan:
        selected_plan = normalize_task_plan(llm_plan.get("task_plan", []))
        selected_skills = llm_plan.get("skills", [])
        active_skills, selected_overrides, selected_skill_plan = _activate_selected_skills(
            active_skills,
            selected_skills if isinstance(selected_skills, list) else [],
        )
        for expert_name, overrides in selected_overrides.items():
            skill_overrides.setdefault(expert_name, {}).update(overrides)
        if selected_skill_plan:
            selected_plan = selected_skill_plan

        direct_answer = str(llm_plan.get("direct_answer", "")).strip()
        if not selected_plan and direct_answer:
            return _build_direct_response(user_query, memory_text, direct_answer=direct_answer)

        decision_plan = _plan_from_module_decisions(llm_plan.get("module_decisions", {}))
        if decision_plan is not None:
            selected_plan = decision_plan

        task_plan = selected_plan
        if not task_plan:
            return _build_direct_response(user_query, memory_text)

        module_tasks = {agent: default_task_for_agent(agent, user_query) for agent in task_plan}
        reason = str(llm_plan.get("reason", "")) or reason
        module_tasks.update(normalize_module_tasks(llm_plan.get("module_tasks", {}), task_plan, user_query))
    elif not task_plan:
        return _build_direct_response(user_query, memory_text)

    logger.info("[agent] supervisor planned modules=%s reason=%s", task_plan, reason[:100])
    return _build_module_plan_update(
        task_plan=task_plan,
        module_tasks=module_tasks,
        reason=reason,
        active_skills=active_skills,
        skill_overrides=skill_overrides,
    )


def _load_active_skill_overrides(active_skills: List[str]) -> tuple[Dict[str, dict], List[str]]:
    """Load already-active skills and return expert overrides plus routing hints."""
    skill_overrides: Dict[str, dict] = {}
    skill_plan: List[str] = []
    if not conf.ENABLE_SKILLS or not active_skills:
        return skill_overrides, skill_plan

    loader = get_skill_loader(conf.SKILLS_DIR)
    for skill_name in active_skills:
        skill = loader.get(skill_name)
        if not skill:
            continue
        for expert_name, overrides in skill.expert_overrides.items():
            agent = normalize_agent_name(expert_name)
            if agent and agent != "FINISH":
                skill_overrides.setdefault(agent, {}).update(overrides)
        if skill.routing_hints and not skill_plan:
            skill_plan = normalize_task_plan(skill.routing_hints)
    return skill_overrides, skill_plan


def _activate_selected_skills(
    active_skills: List[str],
    selected_skills: List[str],
) -> tuple[List[str], Dict[str, dict], List[str]]:
    """Validate and activate newly selected skills."""
    skill_overrides: Dict[str, dict] = {}
    skill_plan: List[str] = []
    if not conf.ENABLE_SKILLS or not selected_skills:
        return active_skills, skill_overrides, skill_plan

    loader = get_skill_loader(conf.SKILLS_DIR)
    for raw_name in selected_skills:
        skill_name = str(raw_name or "").strip()
        if not skill_name:
            continue
        skill = loader.get(skill_name)
        if not skill:
            logger.info("[agent] supervisor ignored unknown skill: %s", skill_name)
            continue
        if skill_name not in active_skills:
            active_skills.append(skill_name)
        for expert_name, overrides in skill.expert_overrides.items():
            agent = normalize_agent_name(expert_name)
            if agent and agent != "FINISH":
                skill_overrides.setdefault(agent, {}).update(overrides)
        if skill.routing_hints and not skill_plan:
            skill_plan = normalize_task_plan(skill.routing_hints)
    return active_skills, skill_overrides, skill_plan


def _build_direct_response(
    user_query: str,
    memory_text: str = "",
    direct_answer: str = "",
) -> Dict[str, Any]:
    """Return a direct supervisor answer without invoking expert modules."""
    if not direct_answer:
        direct_answer = (
            "我还没有识别到需要调用学术模块的明确任务。请补充研究主题或目标，"
            "例如“检索 SCMA 相关文献”“总结这篇论文”“分析方法并生成 MATLAB 复现代码”。"
        )
    if memory_text:
        logger.info("[agent] supervisor direct response with memory context")
    else:
        logger.info("[agent] supervisor direct response")
    return {
        "next_agent": "FINISH",
        "task_plan": [],
        "module_tasks": {},
        "messages": [HumanMessage(content=direct_answer, name="supervisor")],
    }


def _build_module_plan_update(
    task_plan: List[str],
    module_tasks: Dict[str, str],
    reason: str,
    active_skills: List[str],
    skill_overrides: Dict[str, dict],
) -> Dict[str, Any]:
    """Build a single supervisor plan update for the module executor."""
    return {
        "next_agent": "module_executor",
        "task_plan": task_plan,
        "module_tasks": module_tasks,
        "current_task": reason,
        "active_skills": active_skills,
        "skill_overrides": skill_overrides,
        "messages": [HumanMessage(content=f"Supervisor 模块计划: {reason}", name="supervisor")],
    }


def _plan_with_llm(
    user_query: str,
    memory_text: str,
    fallback_plan: List[str],
    active_skills: List[str],
) -> Dict[str, Any]:
    """Ask the supervisor LLM to refine a module plan without running tools."""
    llm = ChatOpenAI(
        model=conf.AGENT_SUPERVISOR_MODEL,
        base_url=conf.AGENT_SUPERVISOR_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_SUPERVISOR_API_KEY, conf.AGENT_SUPERVISOR_BASE_URL),
        temperature=conf.AGENT_SUPERVISOR_TEMPERATURE,
        max_tokens=conf.AGENT_SUPERVISOR_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )

    skill_text = ""
    if conf.ENABLE_SKILLS:
        loader = get_skill_loader(conf.SKILLS_DIR)
        skill_meta = loader.list_metadata()
        if skill_meta:
            skill_lines = [f"- {sm['name']}: {sm['description']}" for sm in skill_meta]
            skill_text = "\n可用技能:\n" + "\n".join(skill_lines)

    prompt = (
        f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
        "你现在只做统一调度规划，不要调用工具，不要执行专家任务。\n"
        "请以用户当前任务为准按需选择模块，允许只选择一个模块；不要因为存在依赖链就自动选择全部模块。\n"
        "请判断是否需要调用模块，并输出严格 JSON。\n\n"
        f"用户请求:\n{user_query or '(empty)'}\n\n"
        f"会话记忆:\n{memory_text or '(none)'}\n\n"
        f"静态候选计划: {fallback_plan}\n"
        f"已激活技能: {active_skills}\n"
        f"{skill_text}\n\n"
        "输出格式:\n"
        "{"
        '"task_plan":["code_builder"],'
        '"module_decisions":{'
        '"literature":{"selected":false,"reason":"已有明确论文或不需要查找候选论文"},'
        '"summarizer":{"selected":true,"reason":"需要总结论文"},'
        '"methodology":{"selected":true,"reason":"需要提取方法规格"},'
        '"code_builder":{"selected":true,"reason":"需要生成并验证复现代码"},'
        '"database_manager":{"selected":false,"reason":"不需要管理本地数据库"},'
        '"writing_editor":{"selected":false,"reason":"不需要论文写作或润色"}'
        "},"
        '"module_tasks":{"code_builder":"..."},'
        '"reason":"...",'
        '"skills":[],' 
        '"direct_answer":""'
        "}\n"
        "请先逐一填写 module_decisions，再让 task_plan 只包含 selected=true 的模块，顺序按信息依赖排列。"
        "如果用户已经给出明确论文标题、paper_id、本地文件或上传论文，且目标是总结/方法/代码，"
        "通常不需要 literature；summarizer 和 methodology 会自行读取指定论文内容。"
        "paper_id、前序专家选择和会话记忆中的 recent_paper_ids 是结构化论文上下文；"
        "自然语言任务只用于决定模块和输出目标，不应直接作为数据库搜索 query。"
        "只有需要查找候选论文、扩展调研、推荐论文、下载论文或补充缺失论文来源时才选择 literature。"
        "如果用户要求删除、导入、列出、搜索、去重、回填本地论文数据库，必须选择 database_manager，不能 direct_answer 声称已执行。"
        "如果用户要求论文写作、学术润色、中英文改写、翻译，或处理 .docx/.tex 文稿，选择 writing_editor。"
        "如果不需要调用学术模块，task_plan 设为 []，direct_answer 给出简短直接回复或澄清问题。"
    )
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
    except Exception as exc:
        logger.info("[agent] supervisor planner LLM failed, using static plan: %s", exc)
        return {}

    text = str(response.content).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _plan_from_module_decisions(raw_decisions: Any) -> List[str] | None:
    """Derive the executable plan from explicit per-module LLM decisions."""
    if not isinstance(raw_decisions, dict) or not raw_decisions:
        return None
    plan: List[str] = []
    saw_decision = False
    for agent in ("literature", "summarizer", "methodology", "code_builder", "database_manager", "writing_editor"):
        decision = raw_decisions.get(agent)
        if not isinstance(decision, dict):
            continue
        saw_decision = True
        if decision.get("selected") is True:
            plan.append(agent)
    return plan if saw_decision else None
