"""Code Builder Expert — generates runnable implementation code from
methodology specifications."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.progress import emit_progress
from scholar_agent.agents.utils import format_memory_context
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.core.runtime import shutdown_requested
from scholar_agent.prompts import CODE_BUILDER_SYSTEM_PROMPT, CODE_REVIEW_SYSTEM_PROMPT
from scholar_agent.skills.loader import get_skill_loader
from scholar_agent.tools import code_tools
from scholar_agent.tools.base import execute_tool_safely
from scholar_agent.tools.registry import get_tool_registry

logger = get_logger(__name__)

_CODE_WORKSPACE_TOOL_NAMES = (
    "code_workspace_write_file",
    "code_workspace_read_file",
    "code_workspace_list_files",
    "code_workspace_make_patch",
    "code_workspace_apply_patch",
    "code_workspace_set_validation_plan",
    "code_workspace_run_python",
    "code_workspace_run_pytest",
    "code_workspace_run_shell",
    "code_workspace_record_validation",
)


def code_builder_node(state: MultiAgentState) -> Dict[str, Any]:
    """Code Builder: generate runnable code from methodology specifications."""
    logger.info("[agent] code_builder start")

    llm = ChatOpenAI(
        model=conf.AGENT_CODE_BUILDER_MODEL,
        base_url=conf.AGENT_CODE_BUILDER_BASE_URL,
        api_key=conf.resolve_api_key(conf.AGENT_CODE_BUILDER_API_KEY, conf.AGENT_CODE_BUILDER_BASE_URL),
        temperature=conf.AGENT_CODE_BUILDER_TEMPERATURE,
        max_tokens=conf.AGENT_CODE_BUILDER_MAX_TOKENS,
        timeout=conf.AGENT_LLM_TIMEOUT,
    )

    task = state.get("current_task", "")
    workspace_path = _configure_code_workspace(state)

    # Gather methodology output from previous expert
    methodology_output = ""
    for eo in state.get("expert_outputs", []):
        if eo.get("expert_name") == "methodology":
            methodology_output = eo.get("content", "")
            break

    context = ""
    if methodology_output:
        context = f"\n\n=== 方法分析专家输出 ===\n{methodology_output}"
    else:
        context = "\n\n注意: 方法分析专家未提供输出，请根据任务描述直接生成代码。"

    # Attempt to extract a paper title from expert outputs
    title = conf.DEFAULT_PAPER_TITLE
    for eo in state.get("expert_outputs", []):
        meta = eo.get("metadata", {})
        if meta.get("paper_ids"):
            title = f"Paper {meta['paper_ids'][0]}"
            break
    project_path, project_slug = _select_code_project(state, workspace_path, task, title, methodology_output)
    python_executable = _configure_code_python_executable(state)

    system_prompt = CODE_BUILDER_SYSTEM_PROMPT.replace("{标题}", title)

    # Apply skill overrides (e.g., language hint from MATLAB skill)
    overrides = state.get("skill_overrides", {}).get("code_builder", {})
    if overrides.get("language_hint"):
        lang = overrides["language_hint"]
        system_prompt += f"\n\n**语言偏好**: 请优先使用 {lang} 编写代码。"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请根据方法分析生成可运行代码。\n\n任务: {task}{format_memory_context(state)}\n{context}"),
    ]

    tool_trace = ""
    tool_call_count = 0
    available_tools: List[str] = []
    delivery_status = "plain_generation"
    if conf.AGENT_CODE_BUILDER_AUTONOMOUS:
        with code_tools.use_code_workspace_dir(project_path), code_tools.use_code_python_executable(python_executable):
            code_tools.clear_code_validation_state()
            emit_progress(
                "expert_detail",
                agent="code_builder",
                detail=f"已选择代码项目目录: {project_path}",
            )
            content, tool_trace, tool_call_count, available_tools, delivery_status = _run_autonomous_coding_loop(
                llm,
                system_prompt,
                task,
                methodology_output,
                state,
                project_path,
                project_slug,
            )
    else:
        try:
            emit_progress(
                "llm_call",
                agent="code_builder",
                label="代码生成",
                detail="正在生成代码实现。",
            )
            response = llm.invoke(messages)
            content = str(response.content).strip()
        except Exception as exc:
            logger.info(f"[agent] code_builder LLM failed: {exc}")
            content = f"代码生成失败: {exc}"

    review = ""
    revised = False
    validation_record = code_tools.get_code_validation_record()
    if (
        delivery_status.startswith("plain_generation")
        and methodology_output
        and content
        and not content.startswith("代码生成失败")
    ):
        emit_progress(
            "llm_call",
            agent="code_builder",
            label="代码审查",
            detail="正在审查生成代码是否满足可运行复现要求。",
        )
        review = _review_code_output(llm, task, methodology_output, content, state)
        if review and "NEEDS_REVISION" in review.upper():
            emit_progress(
                "llm_call",
                agent="code_builder",
                label="代码修订",
                detail="审查要求修改，正在生成修订版。",
            )
            revised_content = _revise_code_output(llm, system_prompt, task, methodology_output, content, review, state)
            if revised_content:
                content = revised_content
                revised = True

    logger.info(f"[agent] code_builder done output={len(content)} chars")
    return {
        "next_agent": "supervisor",
        "expert_outputs": [{
            "expert_name": "code_builder",
            "content": content,
            "metadata": {
                "has_methodology_input": bool(methodology_output),
                "quality_reviewed": bool(review),
                "revised_after_review": revised,
                "autonomous_coding": conf.AGENT_CODE_BUILDER_AUTONOMOUS,
                "autonomous_tool_calls": tool_call_count,
                "code_workspace_path": workspace_path,
                "code_project_path": project_path,
                "code_project_slug": project_slug,
                "delivery_status": delivery_status,
                "validation_evidence": _format_validation_record(validation_record),
                "available_tools": available_tools[:80],
                "tool_trace": tool_trace[:4000],
            },
        }],
        "messages": [
            HumanMessage(content=f"[Code Builder 输出]\n{content[:3000]}", name="code_builder")
        ],
    }


def _run_autonomous_coding_loop(
    llm: ChatOpenAI,
    system_prompt: str,
    task: str,
    methodology_output: str,
    state: MultiAgentState,
    project_path: str,
    project_slug: str,
) -> Tuple[str, str, int, List[str], str]:
    registry = get_tool_registry()
    workspace_path = code_tools.get_code_workspace_dir()
    langchain_tools = []
    tool_map = {}
    available_tool_names = _available_code_builder_tool_names(registry)
    logger.info("[agent] code_builder available tools=%s", available_tool_names)
    for name in available_tool_names:
        tool = registry.get(name)
        if tool is None:
            continue
        langchain_tools.append(registry.to_langchain_tool(name))
        tool_map[name] = tool
    if not langchain_tools:
        logger.info("[agent] code_builder autonomous mode unavailable: no code tools registered")
        return _invoke_plain_code_builder(llm, system_prompt, task, methodology_output, state), "", 0, available_tool_names, "plain_generation"

    if not hasattr(llm, "bind_tools"):
        logger.info("[agent] code_builder autonomous mode unavailable: LLM has no bind_tools")
        return _invoke_plain_code_builder(llm, system_prompt, task, methodology_output, state), "", 0, available_tool_names, "plain_generation"

    tool_llm = llm.bind_tools(langchain_tools)
    skill_context = _format_code_builder_skill_context(state)
    messages: List[Any] = [
        SystemMessage(
            content=(
                f"{system_prompt}\n\n"
                f"你现在具备受控项目级 patch 编码能力。用户选择的代码工作区是: {workspace_path}。\n"
                f"本轮工具根目录是: {project_path}，项目名: {project_slug}。\n"
                "所有工具调用都已经被限制在本轮工具根目录内。工具参数 path 一律使用相对路径，"
                "不要在相对路径中重复项目目录名，也不要额外创建同名或 paper_reproduction 子目录。\n"
                "如果本轮是新建复现项目，项目文件应直接写入当前工具根目录；"
                "如果本轮是查看或回答已有项目问题，先列出和读取当前工具根目录下的相关文件，不要创建文件。"
                "当任务确实需要代码交付时，必须创建 README/运行入口/核心函数或脚本/必要测试或验证脚本。"
                "修改已有文件时优先使用 code_workspace_make_patch 生成 diff，再用 code_workspace_apply_patch 应用；"
                "使用合适的内置或外部 MCP 工具实际运行、测试或检查代码，并根据失败结果修订。"
                "在开始验证前，必须调用 code_workspace_set_validation_plan 声明本次交付必须通过的验证目标；"
                "如果用户或项目文件指定了多个验证脚本/测试/入口，验证计划必须逐项包含它们。"
                "code_workspace_run_python 和 code_workspace_run_pytest 成功时会自动记录验证证据；"
                "需要安装依赖、运行项目 CLI 或执行非 Python 命令时，可以使用受控的 code_workspace_run_shell；"
                "使用外部 MCP 工具验证后，调用 code_workspace_record_validation 记录 tool、target、passed 和 evidence。"
                "只有最后一次文件修改之后，验证计划中的每个目标都有 passed=true 的验证记录，才可以输出完成交付。"
                "不要请求任意 shell，不要写入当前复现项目目录之外的路径。"
                "如果本轮选择创建或修改文件，必须声明验证计划并完成全部验证目标后再输出最终交付说明。"
                "如果本轮用户只是询问、解释、检查或定位已有项目中的信息，可以只读取必要文件并直接回答；"
                "这种只读回答不得写文件、不得声明已运行验证，也不需要创建验证计划。"
                "完成后输出最终说明，包含文件列表、关键 patch、运行命令、测试结果和代码假设。"
                f"{skill_context}"
            )
        ),
        HumanMessage(
            content=(
                f"任务: {task}{format_memory_context(state)}\n\n"
                f"=== 方法分析专家输出 ===\n{methodology_output or '未提供方法分析，请生成最小可运行示例并标注假设。'}"
            )
        ),
    ]

    trace_blocks: List[str] = []
    tool_call_count = 0
    created_or_modified = False
    validated_after_write = False
    validation_plan_set = False
    logger.info("[agent] code_builder tool loop unbounded")
    while True:
        if shutdown_requested():
            logger.info("[agent] code_builder stopping because shutdown was requested")
            return (
                "代码构建已中止：服务正在关闭，未完成最终交付。",
                "\n\n".join(trace_blocks),
                tool_call_count,
                available_tool_names,
                "cancelled",
            )
        try:
            emit_progress(
                "llm_call",
                agent="code_builder",
                label="代码构建",
                detail="正在让模型规划下一步文件修改或验证工具调用。",
            )
            response = tool_llm.invoke(messages)
        except Exception as exc:
            logger.info("[agent] code_builder autonomous invoke failed: %s", exc)
            return (
                _invoke_plain_code_builder(llm, system_prompt, task, methodology_output, state),
                "\n".join(trace_blocks),
                tool_call_count,
                available_tool_names,
                "plain_generation_after_tool_error",
            )

        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            if not created_or_modified:
                if methodology_output and tool_call_count == 0:
                    logger.info("[agent] code_builder final deferred for code delivery before any file inspection/write")
                    messages.append(
                        HumanMessage(
                            content=(
                                "用户任务包含方法规格，通常需要交付代码项目。"
                                "请先读取或列出项目文件，并根据任务决定是否创建或修改真实文件。"
                                "如果你判断本轮不应写文件，必须先读取足够证据再回答。"
                            )
                        )
                    )
                    continue
                logger.info("[agent] code_builder returned read-only/factual final content tool_calls=%s", tool_call_count)
                return (
                    str(response.content or "").strip(),
                    "\n\n".join(trace_blocks),
                    tool_call_count,
                    available_tool_names,
                    "complete_no_write",
                )
            if not _code_project_ready(created_or_modified, validation_plan_set, validated_after_write):
                logger.info(
                    "[agent] code_builder final deferred created=%s validation_plan=%s validated_after_write=%s",
                    created_or_modified,
                    validation_plan_set,
                    validated_after_write,
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "你已经创建或修改了项目文件，但当前复现项目尚未达到交付条件。请继续调用工具完成实现："
                            "调用 code_workspace_set_validation_plan 声明验证目标，"
                            "并在最后一次文件修改后运行验证计划中的每个目标。"
                            "如果任一目标失败，必须读取错误、定位文件和代码行，修改后重新运行完整验证计划。"
                            "必须把文件写入当前项目目录的相对路径，例如 README.md、main.py、tests/test_smoke.py。"
                        )
                    )
                )
                continue
            logger.info("[agent] code_builder autonomous returned final content tool_calls=%s", tool_call_count)
            return (
                str(response.content or "").strip(),
                "\n\n".join(trace_blocks),
                tool_call_count,
                available_tool_names,
                "complete",
            )

        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            if name not in tool_map:
                result_text = f"Tool {name} is not allowed for code_builder."
                messages.append(ToolMessage(content=result_text, tool_call_id=call_id or name))
                continue
            if not isinstance(args, dict):
                args = {}
            emit_progress("tool_start", agent="code_builder", tool=name, input=args)
            result = execute_tool_safely(tool_map[name], args)
            tool_call_count += 1
            if result.status == "success" and name in {
                "code_workspace_write_file",
                "code_workspace_apply_patch",
            }:
                created_or_modified = True
                validated_after_write = False
                code_tools.clear_code_validation_record()
            if result.status == "success" and name == "code_workspace_set_validation_plan":
                validation_plan_set = bool(code_tools.get_code_validation_plan())
                validated_after_write = False
            if name in {
                "code_workspace_run_python",
                "code_workspace_run_pytest",
                "code_workspace_run_shell",
                "code_workspace_record_validation",
            }:
                validated_after_write = bool(created_or_modified and code_tools.code_validation_ready())
            logger.info("[agent] code_builder tool %s status=%s", name, result.status)
            emit_progress(
                "tool_done",
                agent="code_builder",
                tool=name,
                status=result.status,
                summary=str(result.result)[:300],
            )
            result_text = f"[status={result.status}] {result.result}"
            state_text = _format_project_state_for_llm(
                created_or_modified=created_or_modified,
                validation_plan_set=validation_plan_set,
                validated_after_write=validated_after_write,
                validation_record=code_tools.get_code_validation_record(),
                validation_plan=code_tools.get_code_validation_plan(),
                validation_records=code_tools.get_code_validation_records(),
            )
            trace_blocks.append(f"### {name}\nInput: {args}\n{result_text}")
            messages.append(ToolMessage(content=f"{result_text}\n\n{state_text}"[:6000], tool_call_id=call_id or name))

    final_prompt = HumanMessage(
        content=(
            "请基于目前已创建的文件和运行结果，"
            f"输出最终代码交付说明、仍未解决的问题和下一步验证建议。"
            f"交付状态: files_created={created_or_modified}, validation_plan_set={validation_plan_set}, "
            f"validated_after_last_write={validated_after_write}。"
            "如果未在最后一次文件修改后完成全部验证计划，必须明确标注为未完成而不是声称可运行。"
        )
    )
    messages.append(final_prompt)
    try:
        response = llm.invoke(messages)
        content = str(response.content).strip()
    except Exception as exc:
        logger.info("[agent] code_builder final summary failed: %s", exc)
        content = "代码生成已执行工具调用，但最终说明生成失败。"
    return content, "\n\n".join(trace_blocks), tool_call_count, available_tool_names, "complete"


def _available_code_builder_tool_names(registry=None) -> List[str]:
    """Return Code Builder's internal workspace tools plus all external MCP tools."""
    registry = registry or get_tool_registry()
    names: List[str] = []
    for name in _CODE_WORKSPACE_TOOL_NAMES:
        if registry.get(name) is not None:
            names.append(name)

    for name in registry.list_all():
        if name in names:
            continue
        tool = registry.get(name)
        if tool is not None and bool(getattr(tool, "is_external_mcp_tool", False)):
            names.append(name)
    return names


def _recent_code_project_path(state: MultiAgentState, workspace_path: str) -> str:
    memory = state.get("memory") or {}
    recent_path = str(memory.get("recent_code_project_path", "")).strip()
    if not recent_path:
        return ""
    try:
        workspace_root = Path(workspace_path).resolve()
        candidate = Path(recent_path).resolve()
        if candidate == workspace_root or workspace_root in candidate.parents:
            return str(candidate)
    except Exception as exc:
        logger.info("[agent] code_builder ignored recent project path %r: %s", recent_path, exc)
    return ""


def _code_project_ready(created_or_modified: bool, validation_plan_set: bool, validated: bool) -> bool:
    return bool(created_or_modified and validation_plan_set and validated)


def _format_validation_record(record: dict | None) -> str:
    if not isinstance(record, dict) or not record:
        return ""
    return (
        f"passed={record.get('passed')}; "
        f"tool={record.get('tool', '')}; "
        f"target={record.get('target', '')}; "
        f"evidence={record.get('evidence', '')}"
    )


def _format_project_state_for_llm(
    created_or_modified: bool,
    validation_plan_set: bool,
    validated_after_write: bool,
    validation_record: dict | None,
    validation_plan: list[str] | None,
    validation_records: dict[str, dict] | None,
) -> str:
    status = (
        "项目状态: files_created_or_modified="
        f"{bool(created_or_modified)}, validation_plan_set={bool(validation_plan_set)}, "
        f"validated_after_last_write={bool(validated_after_write)}."
    )
    if validation_plan:
        status += "\n验证计划:\n" + "\n".join(f"- {target}" for target in validation_plan)
        records = validation_records or {}
        pending = [
            target
            for target in validation_plan
            if not isinstance(records.get(target), dict) or records[target].get("passed") is not True
        ]
        if pending:
            status += "\n尚未通过的验证目标:\n" + "\n".join(f"- {target}" for target in pending)
    else:
        status += "\n验证计划: 未声明。下一步应调用 code_workspace_set_validation_plan。"
    evidence = _format_validation_record(validation_record)
    if evidence:
        status += f"\n最近验证: {evidence}"
    if _code_project_ready(created_or_modified, validation_plan_set, validated_after_write):
        status += (
            "\n交付条件已满足。下一步应输出最终交付说明；"
            "只有在你基于验证输出发现了具体缺陷时，才继续修改或重新运行验证。"
        )
    else:
        status += "\n交付条件尚未满足。请继续实现、修复或验证。"
    return status


def _format_validation_status() -> str:
    plan = code_tools.get_code_validation_plan()
    records = code_tools.get_code_validation_records()
    if not plan:
        return ""
    lines = []
    for target in plan:
        record = records.get(target)
        if not isinstance(record, dict):
            lines.append(f"- {target}: pending")
            continue
        passed = "passed" if record.get("passed") is True else "failed"
        lines.append(f"- {target}: {passed}; tool={record.get('tool', '')}; evidence={record.get('evidence', '')[:240]}")
    return "\n".join(lines)


def _format_project_inventory(project_path: str, limit: int = 80) -> str:
    root = Path(project_path)
    if not root.exists():
        return ""
    files = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            try:
                rel = item.relative_to(root)
            except ValueError:
                continue
            files.append(f"- {rel.as_posix()} ({item.stat().st_size} bytes)")
            if len(files) >= limit:
                break
    return "\n".join(files)


def _derive_code_project_slug(state: MultiAgentState, task: str, title: str) -> str:
    source = ""
    for output in state.get("expert_outputs", []) or []:
        metadata = output.get("metadata", {}) or {}
        paper_ids = metadata.get("paper_ids", []) or []
        if paper_ids:
            source = f"paper {paper_ids[0]}"
            break
    if not source:
        source = title if title and title != conf.DEFAULT_PAPER_TITLE else task
    raw = str(source or "paper reproduction").strip()
    ascii_text = raw.encode("ascii", errors="ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    if not slug:
        slug = "paper_reproduction"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
    if not slug.endswith(digest):
        slug = f"{slug[:64].strip('_')}_{digest}"
    return slug


def _select_code_project(
    state: MultiAgentState,
    workspace_path: str,
    task: str,
    title: str,
    methodology_output: str,
) -> Tuple[str, str]:
    explicit_project = _explicit_project_path(workspace_path, bool(state.get("code_workspace_is_project", False)))
    if explicit_project:
        candidate = Path(explicit_project).resolve()
        logger.info("[agent] code_builder using explicit project path=%s", candidate)
        return str(candidate), candidate.name

    recent_path = _recent_code_project_path(state, workspace_path)
    recent_status = str((state.get("memory") or {}).get("recent_code_delivery_status", "")).strip()
    if not methodology_output and recent_path and recent_status in {"incomplete", "cancelled"}:
        candidate = Path(recent_path).resolve()
        logger.info("[agent] code_builder continuing recent project=%s", candidate)
        return str(candidate), candidate.name

    project_slug = _derive_code_project_slug(state, task, title)
    return str((Path(workspace_path) / project_slug).resolve()), project_slug


def _explicit_project_path(workspace_path: str, selected_as_project: bool = False) -> str:
    """Return a project root when the provided workspace is itself a concrete project."""
    try:
        path = Path(workspace_path).resolve()
    except Exception:
        return ""
    if not path.exists() or not path.is_dir():
        return ""
    if selected_as_project:
        return str(path)
    try:
        default_workspace = Path(conf.CODE_BUILDER_WORKSPACE_DIR).resolve()
    except Exception:
        default_workspace = None
    if default_workspace is not None and path == default_workspace:
        return ""
    return str(path)


def _invoke_plain_code_builder(
    llm: ChatOpenAI,
    system_prompt: str,
    task: str,
    methodology_output: str,
    state: MultiAgentState,
) -> str:
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"请根据方法分析生成可运行代码。\n\n任务: {task}{format_memory_context(state)}\n"
                        f"\n\n=== 方法分析专家输出 ===\n{methodology_output}"
                    )
                ),
            ]
        )
        return str(response.content).strip()
    except Exception as exc:
        logger.info(f"[agent] code_builder LLM failed: {exc}")
        return f"代码生成失败: {exc}"


def _format_code_builder_skill_context(state: MultiAgentState) -> str:
    active_skills = state.get("active_skills", []) or []
    if not conf.ENABLE_SKILLS or not active_skills:
        return ""
    loader = get_skill_loader(conf.SKILLS_DIR)
    blocks = []
    for name in active_skills:
        skill = loader.get(str(name))
        if not skill:
            continue
        text = (skill.system_prompt_append or skill.description or "").strip()
        if text:
            blocks.append(f"## Skill: {skill.name}\n{text[:2500]}")
    if not blocks:
        return ""
    return "\n\n## 可用技能上下文\n" + "\n\n".join(blocks)


def _configure_code_workspace(state: MultiAgentState) -> str:
    workspace_path = str(state.get("code_workspace_path") or conf.CODE_BUILDER_WORKSPACE_DIR).strip()
    try:
        with code_tools.use_code_workspace_dir(workspace_path) as resolved:
            pass
    except Exception as exc:
        logger.info("[agent] code_builder workspace path invalid %r: %s", workspace_path, exc)
        with code_tools.use_code_workspace_dir(conf.CODE_BUILDER_WORKSPACE_DIR) as resolved:
            pass
    logger.info("[agent] code_builder workspace=%s", resolved)
    return resolved


def _configure_code_python_executable(state: MultiAgentState) -> str:
    selected = str(state.get("code_python_executable") or conf.CODE_BUILDER_PYTHON_EXECUTABLE).strip()
    try:
        with code_tools.use_code_python_executable(selected) as resolved:
            pass
    except Exception as exc:
        logger.info("[agent] code_builder python path invalid %r: %s", selected, exc)
        with code_tools.use_code_python_executable(conf.CODE_BUILDER_PYTHON_EXECUTABLE) as resolved:
            pass
    logger.info("[agent] code_builder python=%s", resolved)
    return resolved


def _review_code_output(
    llm: ChatOpenAI,
    task: str,
    methodology_output: str,
    code_output: str,
    state: MultiAgentState,
) -> str:
    messages = [
        SystemMessage(content=CODE_REVIEW_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"任务: {task}{format_memory_context(state)}\n\n"
                f"=== 方法分析规格 ===\n{methodology_output[:5000]}\n\n"
                f"=== 待审查代码输出 ===\n{code_output[:9000]}"
            )
        ),
    ]
    try:
        response = llm.invoke(messages)
        return str(response.content).strip()
    except Exception as exc:
        logger.info("[agent] code_builder review failed: %s", exc)
        return ""


def _revise_code_output(
    llm: ChatOpenAI,
    system_prompt: str,
    task: str,
    methodology_output: str,
    code_output: str,
    review: str,
    state: MultiAgentState,
) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "请根据质量审查意见修订代码输出。要求保留完整代码块和运行指南，"
                "修复必须修改项，并明确标注论文未说明的 Assumed/TODO。\n\n"
                f"任务: {task}{format_memory_context(state)}\n\n"
                f"=== 方法分析规格 ===\n{methodology_output[:5000]}\n\n"
                f"=== 原代码输出 ===\n{code_output[:9000]}\n\n"
                f"=== 质量审查意见 ===\n{review[:3000]}"
            )
        ),
    ]
    try:
        response = llm.invoke(messages)
        return str(response.content).strip()
    except Exception as exc:
        logger.info("[agent] code_builder revision failed: %s", exc)
        return ""
