"""Literature Searcher Expert — searches local DB and arXiv for related papers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.utils import (
    extract_user_query,
    format_memory_context,
    memory_augmented_query,
)
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.prompts import LITERATURE_SYSTEM_PROMPT
from scholar_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)

_LITERATURE_TOOL_NAMES = (
    "list_local_database",
    "search_local_database",
    "search_arxiv_papers",
    "search_local_papers_chunks",
)


def _run_literature_tool_calls(
    llm: ChatOpenAI,
    registry,
    user_msg: str,
    task: str,
    state: MultiAgentState,
) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """Let the literature LLM choose tool arguments from tool schemas."""
    langchain_tools = []
    for name in _LITERATURE_TOOL_NAMES:
        if registry.get(name) is not None:
            langchain_tools.append(registry.to_langchain_tool(name))
    if not langchain_tools:
        return {}, [], {}

    tool_llm = llm.bind_tools(langchain_tools)
    emit_progress(
        "llm_call",
        agent="literature",
        label="文献检索",
        detail="正在让模型根据工具 schema 选择检索工具和参数。",
    )
    tool_prompt = (
        "你是文献检索专家。请根据用户原始研究主题选择并调用可用工具。\n"
        "调用搜索工具时，严格遵守工具参数 schema 和 description：query 只传研究主题、论文标题、paper_id "
        "或简洁学术关键词；不要传模块计划、工作流说明、总结/分析/生成代码等输出要求。\n"
        "通常需要列出本地库、检索本地相关片段，并在需要外部文献时检索 arXiv。"
    )
    messages = [
        SystemMessage(content=tool_prompt),
        HumanMessage(
            content=(
                f"用户原始请求:\n{user_msg or conf.DEFAULT_LITERATURE_QUERY}\n\n"
                f"当前模块任务，仅作上下文，不要直接作为 query 传给搜索工具:\n{task or '(none)'}"
                f"{format_memory_context(state)}"
            )
        ),
    ]

    try:
        response = tool_llm.invoke(messages)
    except Exception as exc:
        logger.info("[agent] literature tool-calling failed: %s", exc)
        return {}, [], {}

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        logger.info("[agent] literature tool-calling returned no tool calls")
        return {}, [], {}

    results: Dict[str, Any] = {}
    result_blocks: List[str] = []
    selected_queries: Dict[str, str] = {}
    arxiv_search_attempted = False
    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
        if name not in _LITERATURE_TOOL_NAMES:
            continue
        tool = registry.get(name)
        if tool is None:
            continue
        if not isinstance(args, dict):
            args = {}
        if name == "search_arxiv_papers":
            if arxiv_search_attempted:
                logger.info("[agent] literature skipping extra arXiv tool call in same run")
                continue
            arxiv_search_attempted = True
        emit_progress(
            "tool_start",
            agent="literature",
            tool=name,
            input=args,
        )
        try:
            result = tool.execute(**args)
        except Exception as exc:
            logger.info("[agent] literature tool %s failed: %s", name, exc)
            emit_progress(
                "tool_done",
                agent="literature",
                tool=name,
                status="fail",
                summary=str(exc)[:300],
            )
            continue
        query_arg = str(args.get("query") or "").strip()
        if query_arg:
            if name in {"search_local_database", "search_local_papers_chunks"}:
                selected_queries.setdefault("local_query", query_arg)
                selected_queries.setdefault("retrieval_query", query_arg)
            elif name == "search_arxiv_papers":
                selected_queries.setdefault("arxiv_query", query_arg)
                selected_queries.setdefault("retrieval_query", query_arg)
        results[name] = result
        emit_progress(
            "tool_done",
            agent="literature",
            tool=name,
            status=result.status,
            summary=str(result.result)[:300],
        )
        result_blocks.append(
            f"### Tool: {name}\nInput: {args}\nStatus: {result.status}\nResult:\n{result.result}"
        )
        if name == "search_arxiv_papers" and result.status == "fail" and "429" in str(result.result):
            logger.info("[agent] literature arXiv rate-limited; suppressing further arXiv calls")
            arxiv_search_attempted = True
    return results, result_blocks, selected_queries


def literature_node(state: MultiAgentState) -> Dict[str, Any]:
    """Literature Searcher: search local DB and arXiv for related papers."""
    logger.info("[agent] literature start")

    llm = ChatOpenAI(
        model=conf.AGENT_LITERATURE_MODEL,
        base_url=conf.AGENT_LITERATURE_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_LITERATURE_API_KEY, conf.AGENT_LITERATURE_BASE_URL),
        temperature=conf.AGENT_LITERATURE_TEMPERATURE,
        max_tokens=conf.AGENT_LITERATURE_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )

    task = state.get("current_task", "")
    user_msg = extract_user_query(state)
    query = memory_augmented_query(
        state,
        primary_query=user_msg,
        fallback_query=conf.DEFAULT_LITERATURE_QUERY,
    )

    registry = get_tool_registry()

    tool_results, result_blocks, selected_queries = _run_literature_tool_calls(llm, registry, user_msg, task, state)

    list_result = tool_results.get("list_local_database")
    local_papers = list_result.data if list_result and list_result.status == "success" else []
    chunks_result = tool_results.get("search_local_papers_chunks")
    chunks = chunks_result.data if chunks_result and chunks_result.status == "success" else []
    paper_ids = _paper_ids_from_chunks(chunks)

    if not result_blocks:
        # Fallback for models/backends without tool-call support. Keep this path
        # simple: use the user's original request as query, never the module task.
        list_tool = registry.get("list_local_database")
        emit_progress("tool_start", agent="literature", tool="list_local_database", input={})
        list_result = list_tool.execute()
        emit_progress(
            "tool_done",
            agent="literature",
            tool="list_local_database",
            status=list_result.status,
            summary=str(list_result.result)[:300],
        )
        local_papers = list_result.data if list_result.status == "success" else []

        arxiv_tool = registry.get("search_arxiv_papers")
        try:
            emit_progress(
                "tool_start",
                agent="literature",
                tool="search_arxiv_papers",
                input={"query": query, "max_results": conf.EXPERT_LITERATURE_ARXIV_MAX_RESULTS},
            )
            arxiv_result = arxiv_tool.execute(query=query, max_results=conf.EXPERT_LITERATURE_ARXIV_MAX_RESULTS)
            emit_progress(
                "tool_done",
                agent="literature",
                tool="search_arxiv_papers",
                status=arxiv_result.status,
                summary=str(arxiv_result.result)[:300],
            )
        except Exception as exc:
            logger.info(f"[agent] literature arXiv search failed: {exc}")
            emit_progress(
                "tool_done",
                agent="literature",
                tool="search_arxiv_papers",
                status="fail",
                summary=str(exc)[:300],
            )
            arxiv_result = None

        search_tool = registry.get("search_local_papers_chunks")
        emit_progress(
            "tool_start",
            agent="literature",
            tool="search_local_papers_chunks",
            input={"query": query, "top_k": 5},
        )
        search_result = search_tool.execute(query=query, top_k=5)
        emit_progress(
            "tool_done",
            agent="literature",
            tool="search_local_papers_chunks",
            status=search_result.status,
            summary=str(search_result.result)[:300],
        )
        chunks = search_result.data if search_result.status == "success" else []
        paper_ids = _paper_ids_from_chunks(chunks)
        selected_queries.setdefault("local_query", query)
        selected_queries.setdefault("retrieval_query", query)
        for name, result in (
            ("list_local_database", list_result),
            ("search_arxiv_papers", arxiv_result),
            ("search_local_papers_chunks", search_result),
        ):
            if result is None:
                continue
            result_blocks.append(
                f"### Tool: {name}\nInput: fallback user query\nStatus: {result.status}\nResult:\n{result.result}"
            )

    system_prompt = LITERATURE_SYSTEM_PROMPT.replace("{主题}", query)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"请基于以下检索结果生成文献调研报告。\n\n"
                f"任务上下文: {task}{format_memory_context(state)}\n\n"
                f"### 工具调用结果\n" + "\n\n".join(result_blocks)
            )
        ),
    ]

    try:
        emit_progress(
            "llm_call",
            agent="literature",
            label="文献检索",
            detail="正在基于工具结果生成文献调研报告。",
        )
        response = llm.invoke(messages)
        content = str(response.content).strip()
    except Exception as exc:
        logger.info(f"[agent] literature LLM failed: {exc}")
        raw_report = (
            f"## 📚 文献调研: {query}\n\n"
            f"### 工具调用结果\n" + "\n\n".join(result_blocks)
        )
        content = f"LLM 摘要生成失败，以下是原始检索结果:\n\n{raw_report}"

    logger.info(f"[agent] literature done local={len(local_papers)} arxiv_searched=True output={len(content)} chars")
    return {
        "next_agent": "supervisor",
        "expert_outputs": [{
            "expert_name": "literature",
            "content": content,
            "metadata": {
                "local_paper_count": len(local_papers),
                "arxiv_results": True,
                "paper_ids": paper_ids,
                "chunks_read": len(chunks),
                "retrieval_query": selected_queries.get("retrieval_query", query),
                "local_query": selected_queries.get("local_query", selected_queries.get("retrieval_query", query)),
                "arxiv_query": selected_queries.get("arxiv_query", ""),
            },
        }],
        "messages": [
            HumanMessage(content=f"[Literature Searcher 输出]\n{content[:3000]}", name="literature")
        ],
    }


def _paper_ids_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    paper_ids: List[str] = []
    seen = set()
    for chunk in chunks or []:
        pid = str(chunk.get("paper_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        paper_ids.append(pid)
    return paper_ids
