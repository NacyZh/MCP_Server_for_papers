"""Methodology Analyst Expert — extracts algorithms, math specs, and
experimental setups from paper method sections."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.utils import (
    build_paper_context,
    collect_context_paper_ids,
    collect_paper_ids_from_outputs,
    collect_retrieval_query_from_outputs,
    extract_user_query,
    filter_chunks_by_paper_ids,
    format_memory_context,
    format_prior_expert_context,
    no_papers_response,
    retrieval_query_from_context,
)
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.prompts import METHODOLOGY_SYSTEM_PROMPT
from scholar_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)


def methodology_node(state: MultiAgentState) -> Dict[str, Any]:
    """Methodology Analyst: extract algorithms, math, and experimental details from papers."""
    logger.info("[agent] methodology start")

    llm = ChatOpenAI(
        model=conf.AGENT_METHODOLOGY_MODEL,
        base_url=conf.AGENT_METHODOLOGY_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_METHODOLOGY_API_KEY, conf.AGENT_METHODOLOGY_BASE_URL),
        temperature=conf.AGENT_METHODOLOGY_TEMPERATURE,
        max_tokens=conf.AGENT_METHODOLOGY_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )

    task = state.get("current_task", "")
    user_msg = extract_user_query(state)

    registry = get_tool_registry()
    selected_paper_ids = collect_context_paper_ids(
        state,
        preferred_experts=["summarizer", "literature"],
    )
    if selected_paper_ids:
        chunk_tool = registry.get("get_local_paper_chunks")
        args = {"paper_ids": ",".join(selected_paper_ids[:3]), "max_chunks_per_paper": 8}
        emit_progress("tool_start", agent="methodology", tool="get_local_paper_chunks", input=args)
        search_result = chunk_tool.execute(**args)
        emit_progress(
            "tool_done",
            agent="methodology",
            tool="get_local_paper_chunks",
            status=search_result.status,
            summary=str(search_result.result)[:300],
        )
        chunks = search_result.data if search_result.status == "success" else []
    else:
        search_tool = registry.get("search_local_papers_chunks")
        prior_query = collect_retrieval_query_from_outputs(
            state,
            preferred_experts=["summarizer", "literature"],
        )
        query = retrieval_query_from_context(
            state,
            prior_query=prior_query,
            fallback_query=conf.DEFAULT_SUMMARIZER_QUERY,
        )
        method_query = f"{query} {conf.DEFAULT_METHODOLOGY_QUERY_SUFFIX}"
        emit_progress(
            "tool_start",
            agent="methodology",
            tool="search_local_papers_chunks",
            input={"query": method_query, "top_k": 8},
        )
        search_result = search_tool.execute(query=method_query, top_k=8)
        emit_progress(
            "tool_done",
            agent="methodology",
            tool="search_local_papers_chunks",
            status=search_result.status,
            summary=str(search_result.result)[:300],
        )
        chunks = search_result.data if search_result.status == "success" else []
        selected_paper_ids = collect_paper_ids_from_outputs(
            state,
            preferred_experts=["summarizer", "literature"],
        )
        chunks = filter_chunks_by_paper_ids(chunks, selected_paper_ids)
    if not chunks:
        return no_papers_response("methodology")

    paper_ctx = build_paper_context(chunks, max_chars_per_paper=conf.EXPERT_METHODOLOGY_MAX_CHARS_PER_PAPER)
    system_prompt = METHODOLOGY_SYSTEM_PROMPT.replace("{标题}", paper_ctx["title"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"请分析以下论文的方法部分。优先保持与前序文献筛选和论文总结相同的 paper_id，"
                f"不要混入无关论文的方法。\n\n"
                f"任务上下文: {task}{format_memory_context(state)}"
                f"{format_prior_expert_context(state, ['literature', 'summarizer'])}\n"
                f"{paper_ctx['context']}"
            )
        ),
    ]

    try:
        emit_progress(
            "llm_call",
            agent="methodology",
            label="方法分析",
            detail=f"正在提取 {len(paper_ctx['paper_ids'])} 篇论文的方法规格。",
        )
        response = llm.invoke(messages)
        content = str(response.content).strip()
    except Exception as exc:
        logger.info(f"[agent] methodology LLM failed: {exc}")
        content = f"方法分析失败: {exc}"

    logger.info(f"[agent] methodology done papers={len(paper_ctx['paper_ids'])} output={len(content)} chars")
    return {
        "next_agent": "supervisor",
        "expert_outputs": [{
            "expert_name": "methodology",
            "content": content,
            "metadata": {
                "paper_ids": paper_ctx["paper_ids"],
                "chunks_read": len(chunks),
                "used_prior_selection": bool(selected_paper_ids),
            },
        }],
        "messages": [
            HumanMessage(content=f"[Methodology Analyst 输出]\n{content[:3000]}", name="methodology")
        ],
    }
