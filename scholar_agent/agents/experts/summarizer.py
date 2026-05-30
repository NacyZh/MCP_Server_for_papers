"""Summarizer Expert — reads paper chunks and produces structured summaries."""

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
from scholar_agent.prompts import SUMMARIZER_SYSTEM_PROMPT
from scholar_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)


def summarizer_node(state: MultiAgentState) -> Dict[str, Any]:
    """Summarizer expert: read paper chunks from the DB, produce structured summary."""
    logger.info("[agent] summarizer start")

    llm = ChatOpenAI(
        model=conf.AGENT_SUMMARIZER_MODEL,
        base_url=conf.AGENT_SUMMARIZER_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_SUMMARIZER_API_KEY, conf.AGENT_SUMMARIZER_BASE_URL),
        temperature=conf.AGENT_SUMMARIZER_TEMPERATURE,
        max_tokens=conf.AGENT_SUMMARIZER_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )

    task = state.get("current_task", "")
    user_msg = extract_user_query(state)

    registry = get_tool_registry()
    selected_paper_ids = collect_context_paper_ids(state, preferred_experts=["literature"])
    if selected_paper_ids:
        chunk_tool = registry.get("get_local_paper_chunks")
        args = {"paper_ids": ",".join(selected_paper_ids[:3]), "max_chunks_per_paper": 8}
        emit_progress("tool_start", agent="summarizer", tool="get_local_paper_chunks", input=args)
        search_result = chunk_tool.execute(**args)
        emit_progress(
            "tool_done",
            agent="summarizer",
            tool="get_local_paper_chunks",
            status=search_result.status,
            summary=str(search_result.result)[:300],
        )
        chunks = search_result.data if search_result.status == "success" else []
    else:
        search_tool = registry.get("search_local_papers_chunks")
        prior_query = collect_retrieval_query_from_outputs(state, preferred_experts=["literature"])
        query = retrieval_query_from_context(
            state,
            prior_query=prior_query,
            fallback_query=conf.DEFAULT_SUMMARIZER_QUERY,
        )
        emit_progress(
            "tool_start",
            agent="summarizer",
            tool="search_local_papers_chunks",
            input={"query": query, "top_k": 8},
        )
        search_result = search_tool.execute(query=query, top_k=8)
        emit_progress(
            "tool_done",
            agent="summarizer",
            tool="search_local_papers_chunks",
            status=search_result.status,
            summary=str(search_result.result)[:300],
        )
        chunks = search_result.data if search_result.status == "success" else []
        selected_paper_ids = collect_paper_ids_from_outputs(state, preferred_experts=["literature"])
        chunks = filter_chunks_by_paper_ids(chunks, selected_paper_ids)
    if not chunks:
        logger.info("[agent] summarizer found no papers in local DB")
        return no_papers_response("summarizer")

    paper_ctx = build_paper_context(chunks, max_chars_per_paper=conf.EXPERT_CONTEXT_MAX_CHARS_PER_PAPER)
    system_prompt = SUMMARIZER_SYSTEM_PROMPT.replace("{标题}", paper_ctx["title"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"请总结以下论文。若前序文献检索已经给出候选论文，请保持 paper_id 和引用链一致。\n\n"
                f"任务上下文: {task}{format_memory_context(state)}"
                f"{format_prior_expert_context(state, ['literature'])}\n"
                f"{paper_ctx['context']}"
            )
        ),
    ]

    try:
        emit_progress(
            "llm_call",
            agent="summarizer",
            label="论文总结",
            detail=f"正在总结 {len(paper_ctx['paper_ids'])} 篇论文的相关片段。",
        )
        response = llm.invoke(messages)
        content = str(response.content).strip()
    except Exception as exc:
        logger.info(f"[agent] summarizer LLM failed: {exc}")
        content = f"总结生成失败: {exc}"

    logger.info(f"[agent] summarizer done papers={len(paper_ctx['paper_ids'])} output={len(content)} chars")
    return {
        "next_agent": "supervisor",
        "expert_outputs": [{
            "expert_name": "summarizer",
            "content": content,
            "metadata": {
                "paper_ids": paper_ctx["paper_ids"],
                "chunks_read": len(chunks),
                "used_literature_selection": bool(selected_paper_ids),
            },
        }],
        "messages": [
            HumanMessage(content=f"[Summarizer Expert 输出]\n{content[:3000]}", name="summarizer")
        ],
    }
