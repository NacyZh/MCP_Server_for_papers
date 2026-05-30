"""Writing Editor Expert — academic drafting, translation, and polishing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.utils import format_memory_context, format_prior_expert_context
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.prompts import WRITING_EDITOR_SYSTEM_PROMPT
from scholar_agent.tools.base import execute_tool_safely
from scholar_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)

_WRITING_TOOL_NAMES = (
    "writing_list_documents",
    "writing_read_document",
    "writing_read_docx_document",
    "writing_read_latex_document",
    "writing_write_text_document",
    "writing_write_docx_document",
    "writing_write_latex_document",
    "writing_compile_latex_document",
)


def writing_editor_node(state: MultiAgentState) -> Dict[str, Any]:
    """Run the academic writing editor with controlled document tools."""
    logger.info("[agent] writing_editor start")
    registry = get_tool_registry()
    available_tool_names = [name for name in _WRITING_TOOL_NAMES if registry.get(name) is not None]
    tool_map = {name: registry.get(name) for name in available_tool_names}
    langchain_tools = [registry.to_langchain_tool(name) for name in available_tool_names]

    task = state.get("current_task", "")
    if not langchain_tools:
        content = "论文写作润色未完成：当前没有可用的写作文档工具。"
        return _build_output(content, [], available_tool_names)

    llm = ChatOpenAI(
        model=conf.AGENT_WRITING_EDITOR_MODEL,
        base_url=conf.AGENT_WRITING_EDITOR_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_WRITING_EDITOR_API_KEY, conf.AGENT_WRITING_EDITOR_BASE_URL),
        temperature=conf.AGENT_WRITING_EDITOR_TEMPERATURE,
        max_tokens=conf.AGENT_WRITING_EDITOR_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )
    if not hasattr(llm, "bind_tools"):
        content = _invoke_plain_writing_editor(llm, task, state)
        return _build_output(content, [], available_tool_names)

    tool_llm = llm.bind_tools(langchain_tools)
    prior_context = format_prior_expert_context(
        state,
        ["literature", "summarizer", "methodology"],
        max_chars_per_expert=12000,
    )
    messages: List[Any] = [
        SystemMessage(
            content=(
                f"{WRITING_EDITOR_SYSTEM_PROMPT}\n\n"
                f"受控写作工作区: {conf.WRITING_WORKSPACE_DIR}\n"
                "所有文档工具的 path 都是相对该工作区的路径。不要请求任意 shell。\n"
                "前序专家输出是本轮写作任务的主要素材；不要臆造一个不存在的总结文件路径去读取。"
            )
        ),
        HumanMessage(
            content=(
                f"任务: {task}"
                f"{prior_context}"
                f"{format_memory_context(state)}"
                "\n\n如果任务要求生成文档，必须调用相应写入工具并以工具 success 结果作为完成依据。"
            )
        ),
    ]
    trace: List[Tuple[str, dict, str, str]] = []
    final_content = ""
    tool_call_count = 0

    for _ in range(8):
        emit_progress(
            "llm_call",
            agent="writing_editor",
            label="论文写作润色",
            detail="正在分析写作任务并选择文档工具。",
        )
        try:
            response = tool_llm.invoke(messages)
        except Exception as exc:
            logger.info("[agent] writing_editor LLM failed: %s", exc)
            final_content = f"论文写作润色未完成：模型调用失败: {exc}"
            break
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
                result_text = f"Tool {name} is not allowed for writing_editor."
                messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
                trace.append((name, args, "fail", result_text))
                continue
            emit_progress("tool_start", agent="writing_editor", tool=name, input=args)
            result = execute_tool_safely(tool_map[name], args)
            tool_call_count += 1
            logger.info("[agent] writing_editor tool %s status=%s", name, result.status)
            emit_progress(
                "tool_done",
                agent="writing_editor",
                tool=name,
                status=result.status,
                summary=result.result[:500],
            )
            result_text = f"status={result.status}\n{result.result}"
            messages.append(ToolMessage(content=result_text[:8000], tool_call_id=call_id))
            trace.append((name, args, result.status, result.result))

    successful_writes = _successful_write_calls(trace)
    failed_calls = [item for item in trace if item[2] != "success"]

    if failed_calls and not successful_writes:
        final_content = (
            "论文写作文档生成未完成：本轮没有成功的文档写入工具调用，因此没有生成新的 Word 或 LaTeX 文档。"
            "\n\n"
            f"{_summarize_trace(trace)}"
        )
    elif not final_content:
        final_content = _summarize_trace(trace)
    elif tool_call_count:
        final_content = f"{final_content}\n\n{_summarize_trace(trace)}"

    logger.info("[agent] writing_editor done tool_calls=%s output=%s", tool_call_count, len(final_content))
    return _build_output(final_content, trace, available_tool_names)


def _invoke_plain_writing_editor(llm: ChatOpenAI, task: str, state: MultiAgentState) -> str:
    try:
        emit_progress(
            "llm_call",
            agent="writing_editor",
            label="论文写作润色",
            detail="正在生成写作或润色结果。",
        )
        response = llm.invoke(
            [
                SystemMessage(content=WRITING_EDITOR_SYSTEM_PROMPT),
                HumanMessage(content=f"任务: {task}{format_memory_context(state)}"),
            ]
        )
        return str(response.content).strip()
    except Exception as exc:
        logger.info("[agent] writing_editor plain mode failed: %s", exc)
        return f"论文写作润色未完成：模型调用失败: {exc}"


def _summarize_trace(trace: List[Tuple[str, dict, str, str]]) -> str:
    if not trace:
        return "未调用文档工具；如需读写 .docx 或 .tex，请提供相对 workspace/writing 的文件路径或要求输出文件。"
    lines = ["## 文档工具执行记录"]
    for name, args, status, result in trace:
        lines.append(f"- `{name}` status=`{status}` args=`{args}`")
        lines.append(f"  result: {result}")
    return "\n".join(lines)


def _successful_write_calls(trace: List[Tuple[str, dict, str, str]]) -> List[Tuple[str, dict, str, str]]:
    write_tools = {
        "writing_write_text_document",
        "writing_write_docx_document",
        "writing_write_latex_document",
    }
    return [item for item in trace if item[0] in write_tools and item[2] == "success"]


def _generated_documents(trace: List[Tuple[str, dict, str, str]]) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    for name, args, status, result in _successful_write_calls(trace):
        path = str(args.get("path") or "").strip()
        if not path:
            continue
        documents.append({"path": path, "tool": name, "status": status, "result": result[:500]})
    return documents


def _build_output(
    content: str,
    trace: List[Tuple[str, dict, str, str]],
    available_tool_names: List[str],
) -> Dict[str, Any]:
    return {
        "next_agent": "supervisor",
        "expert_outputs": [
            {
                "expert_name": "writing_editor",
                "content": content,
                "metadata": {
                    "available_tools": available_tool_names,
                    "generated_documents": _generated_documents(trace),
                    "tool_calls": [
                        {"tool": name, "args": args, "status": status, "result": result[:1200]}
                        for name, args, status, result in trace
                    ],
                    "writing_workspace": conf.WRITING_WORKSPACE_DIR,
                },
            }
        ],
        "messages": [HumanMessage(content=f"[Writing Editor 输出]\n{content[:3000]}", name="writing_editor")],
    }
