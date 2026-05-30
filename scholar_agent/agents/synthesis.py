"""Final synthesis node for the modular multi-agent workflow."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.utils import extract_user_query
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.prompts import SUPERVISOR_SYNTHESIS_PROMPT

logger = get_logger(__name__)


def synthesis_node(state: MultiAgentState) -> Dict[str, Any]:
    """Synthesize the final answer once after all selected modules complete."""
    expert_outputs = state.get("expert_outputs", [])
    if not expert_outputs:
        logger.info("[agent] synthesis finished with no expert outputs")
        return {
            "next_agent": "FINISH",
            "messages": [
                HumanMessage(
                    content="未能收集到任何专家输出。请尝试更具体地描述您的问题。",
                    name="supervisor",
                )
            ],
        }

    user_query = extract_user_query(state).strip() or "未指定"
    if expert_outputs and all(eo.get("expert_name") == "database_manager" for eo in expert_outputs):
        final_text = "\n\n".join(str(eo.get("content", "")).strip() for eo in expert_outputs if eo.get("content"))
        logger.info("[agent] synthesis bypassed for database_manager-only result")
        return {
            "next_agent": "FINISH",
            "messages": [HumanMessage(content=final_text or "数据库管理未完成：没有工具执行结果。", name="supervisor")],
        }
    if _has_failed_writing_delivery(expert_outputs):
        final_text = _deterministic_failed_writing_answer(expert_outputs, user_query)
        logger.info("[agent] synthesis bypassed for failed writing delivery")
        return {
            "next_agent": "FINISH",
            "messages": [HumanMessage(content=final_text, name="supervisor")],
        }

    outputs_text = "\n\n---\n\n".join(_format_expert_output(eo) for eo in expert_outputs)

    synthesis_prompt = SUPERVISOR_SYNTHESIS_PROMPT.format(
        expert_outputs=outputs_text,
        user_query=user_query,
    )

    llm = ChatOpenAI(
        model=conf.AGENT_SUPERVISOR_MODEL,
        base_url=conf.AGENT_SUPERVISOR_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_SUPERVISOR_API_KEY, conf.AGENT_SUPERVISOR_BASE_URL),
        temperature=conf.AGENT_SUPERVISOR_SYNTHESIS_TEMPERATURE,
        max_tokens=conf.AGENT_SUPERVISOR_SYNTHESIS_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )

    try:
        emit_progress(
            "llm_call",
            agent="synthesis",
            label="最终整合",
            detail="正在整合所有专家输出并生成最终回复。",
        )
        response = llm.invoke([SystemMessage(content=synthesis_prompt)])
        final_text = str(response.content).strip()
    except Exception as exc:
        logger.info("[agent] synthesis LLM failed: %s", exc)
        final_text = f"专家工作已完成，但整合生成失败: {exc}\n\n{outputs_text}"

    logger.info("[agent] synthesis done chars=%s", len(final_text))
    return {
        "next_agent": "FINISH",
        "messages": [HumanMessage(content=final_text, name="supervisor")],
    }


def _format_expert_output(output: dict) -> str:
    name = output.get("expert_name", "unknown")
    content = str(output.get("content", "") or "")[:conf.SUPERVISOR_SYNTHESIS_CONTENT_TRUNC]
    metadata = output.get("metadata", {}) or {}
    lines = [f"### {name}", content]

    if name == "writing_editor":
        generated = metadata.get("generated_documents", []) or []
        tool_calls = metadata.get("tool_calls", []) or []
        lines.append("\n写作工具事实记录:")
        if generated:
            for doc in generated:
                lines.append(
                    f"- generated path={doc.get('path')} tool={doc.get('tool')} status={doc.get('status')}"
                )
        else:
            lines.append("- generated_documents: none")
        for call in tool_calls:
            lines.append(
                f"- tool={call.get('tool')} status={call.get('status')} args={call.get('args')} result={call.get('result')}"
            )

    return "\n".join(lines)


def _has_failed_writing_delivery(expert_outputs: List[dict]) -> bool:
    writing_outputs = [eo for eo in expert_outputs if eo.get("expert_name") == "writing_editor"]
    if not writing_outputs:
        return False
    for output in writing_outputs:
        content = str(output.get("content", "") or "").strip()
        metadata = output.get("metadata", {}) or {}
        tool_calls = metadata.get("tool_calls", []) or []
        has_write_call = any(str(call.get("tool", "")).startswith("writing_write_") for call in tool_calls)
        has_success_doc = bool(metadata.get("generated_documents"))
        has_failed_write = any(
            str(call.get("tool", "")).startswith("writing_write_") and call.get("status") != "success"
            for call in tool_calls
        )
        if not has_success_doc and (has_failed_write or (has_write_call and not has_success_doc)):
            return True
        if content.startswith("论文写作文档生成未完成") and not has_success_doc:
            return True
    return False


def _deterministic_failed_writing_answer(expert_outputs: List[dict], user_query: str) -> str:
    non_writing = [eo for eo in expert_outputs if eo.get("expert_name") != "writing_editor"]
    writing_outputs = [eo for eo in expert_outputs if eo.get("expert_name") == "writing_editor"]
    summary_parts = [
        str(eo.get("content", "") or "").strip()[:1800]
        for eo in non_writing
        if str(eo.get("content", "") or "").strip()
    ]

    lines = [
        "文档生成未完成。",
        "",
        "根据本轮写作专家的工具调用记录，没有任何 `writing_write_docx_document`、`writing_write_latex_document` 或其他写入工具返回 success，因此不能声称 Word 或 LaTeX 文档已经生成。",
        "",
        "已完成的内容：",
    ]
    if summary_parts:
        lines.append("- 已完成文献检索/总结，下面保留可用于重新生成文档的摘要内容。")
    else:
        lines.append("- 未获得可用于写入文档的有效摘要内容。")

    lines.extend(["", "写作工具调用记录："])
    for output in writing_outputs:
        metadata = output.get("metadata", {}) or {}
        calls = metadata.get("tool_calls", []) or []
        if not calls:
            lines.append("- 未调用写作文档工具。")
        for call in calls:
            lines.append(
                f"- `{call.get('tool')}` status=`{call.get('status')}` args=`{call.get('args')}` result={call.get('result')}"
            )

    if summary_parts:
        lines.extend(["", "可用摘要内容摘录：", "\n\n".join(summary_parts)])

    lines.extend([
        "",
        "下一步建议：重新执行该请求；修复后写作专家会直接使用前序总结内容，并必须通过写入工具成功返回后才报告文档路径。",
        f"",
        f"原始请求：{user_query}",
    ])
    return "\n".join(lines)
