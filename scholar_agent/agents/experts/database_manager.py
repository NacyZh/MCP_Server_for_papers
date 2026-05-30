"""Database Manager Expert — executes local paper database management tools."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.utils import extract_user_query
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.prompts import DATABASE_MANAGER_SYSTEM_PROMPT
from scholar_agent.tools.base import execute_tool_safely
from scholar_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)

_DATABASE_TOOL_NAMES = (
    "list_local_database",
    "search_local_database",
    "add_paper_to_database",
    "delete_paper_from_database",
    "dedup_local_database",
    "backfill_paper_metadata",
    "download_arxiv_papers"
)


def database_manager_node(state: MultiAgentState) -> Dict[str, Any]:
    """Execute requested local database management through registered tools."""
    logger.info("[agent] database_manager start")
    registry = get_tool_registry()
    available_tool_names = [name for name in _DATABASE_TOOL_NAMES if registry.get(name) is not None]
    tool_map = {name: registry.get(name) for name in available_tool_names}
    langchain_tools = [registry.to_langchain_tool(name) for name in available_tool_names]

    task = state.get("current_task", "") or extract_user_query(state)
    if not langchain_tools:
        content = "数据库管理未完成：当前没有可用的本地数据库管理工具。"
        return _build_output(content, [], available_tool_names)

    llm = ChatOpenAI(
        model=conf.AGENT_SUPERVISOR_MODEL,
        base_url=conf.AGENT_SUPERVISOR_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_SUPERVISOR_API_KEY, conf.AGENT_SUPERVISOR_BASE_URL),
        temperature=0,
        max_tokens=conf.AGENT_SUPERVISOR_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )
    if not hasattr(llm, "bind_tools"):
        content = "数据库管理未完成：当前模型接口不支持工具调用，不能执行数据库变更。"
        return _build_output(content, [], available_tool_names)

    tool_llm = llm.bind_tools(langchain_tools)
    messages: List[Any] = [
        SystemMessage(content=DATABASE_MANAGER_SYSTEM_PROMPT),
        HumanMessage(content=f"用户请求:\n{task}"),
    ]

    trace: List[Tuple[str, dict, str, str]] = []
    tool_call_count = 0
    final_content = ""

    for _ in range(4):
        emit_progress(
            "llm_call",
            agent="database_manager",
            label="数据库管理",
            detail="正在判断需要调用的本地数据库工具。",
        )
        response = tool_llm.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            final_content = str(getattr(response, "content", "") or "").strip()
            break

        for call in tool_calls:
            name = str(call.get("name") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            call_id = call.get("id") or name
            if name not in tool_map or tool_map[name] is None:
                result_text = f"Tool {name} is not allowed for database_manager."
                messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
                trace.append((name, args, "fail", result_text))
                continue
            emit_progress("tool_start", agent="database_manager", tool=name, input=args)
            result = execute_tool_safely(tool_map[name], args)
            tool_call_count += 1
            logger.info("[agent] database_manager tool %s status=%s", name, result.status)
            emit_progress(
                "tool_done",
                agent="database_manager",
                tool=name,
                status=result.status,
                summary=result.result[:500],
            )
            messages.append(
                ToolMessage(
                    content=f"status={result.status}\n{result.result}"[:6000],
                    tool_call_id=call_id,
                )
            )
            trace.append((name, args, result.status, result.result))

    if tool_call_count == 0:
        content = "数据库管理未完成：没有实际调用任何数据库工具，因此未执行本地数据库变更。"
    else:
        content = _summarize_database_tool_trace(trace, final_content)

    logger.info("[agent] database_manager done tool_calls=%s", tool_call_count)
    return _build_output(content, trace, available_tool_names)


def _summarize_database_tool_trace(trace: List[Tuple[str, dict, str, str]], final_content: str) -> str:
    lines = ["## 数据库操作结果"]
    for name, args, status, result in trace:
        lines.append(f"- 工具: `{name}`")
        lines.append(f"  参数: `{args}`")
        lines.append(f"  状态: `{status}`")
        lines.append(f"  结果: {result}")
    if final_content:
        lines.append("")
        lines.append("## 模型补充说明")
        lines.append(final_content)
    if not any(status == "fail" for _, _, status, _ in trace):
        return "\n".join(lines)
    return "数据库管理未完成：至少一个工具调用失败。\n\n" + "\n".join(lines)


def _build_output(
    content: str,
    trace: List[Tuple[str, dict, str, str]],
    available_tool_names: List[str],
) -> Dict[str, Any]:
    return {
        "next_agent": "supervisor",
        "expert_outputs": [
            {
                "expert_name": "database_manager",
                "content": content,
                "metadata": {
                    "available_tools": available_tool_names,
                    "tool_calls": [
                        {"tool": name, "args": args, "status": status, "result": result[:1200]}
                        for name, args, status, result in trace
                    ],
                },
            }
        ],
        "messages": [HumanMessage(content=f"[Database Manager 输出]\n{content[:3000]}", name="database_manager")],
    }
