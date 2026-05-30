"""MultiAgentService — wraps the LangGraph graph for the web API."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Any, Dict, Generator, List, Optional

from langchain_core.messages import HumanMessage

from scholar_agent.agents.state import MultiAgentState
from scholar_agent.agents.graph import get_graph
from scholar_agent.agents.progress import reset_progress_sink, set_progress_sink
from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.core.runtime import shutdown_requested
from scholar_agent.storage.memory_store import AgentMemoryStore, get_agent_memory_store

logger = get_logger(__name__)

EXPERT_NAMES = frozenset({
    "summarizer",
    "methodology",
    "code_builder",
    "literature",
    "database_manager",
    "writing_editor",
})

_EXPERT_LABELS = {
    "literature": "文献检索",
    "summarizer": "论文总结",
    "methodology": "方法分析",
    "code_builder": "代码生成",
    "database_manager": "数据库管理",
    "writing_editor": "论文写作润色",
    "synthesis": "最终整合",
}


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _is_final_supervisor_message(message: Any) -> bool:
    """Return True only for final synthesis, not intermediate routing notes."""
    if not isinstance(message, HumanMessage) or getattr(message, "name", None) != "supervisor":
        return False
    content = str(message.content)
    return not content.startswith(("Supervisor 调度:", "Supervisor 模块计划:"))


def _strip_supervisor_control_prefix(content: str) -> str:
    """Remove internal supervisor control prefixes for UI trace labels."""
    text = str(content)
    for prefix in ("Supervisor 调度:", "Supervisor 模块计划:"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _normalize_progress_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert low-level progress events into UI-friendly labels."""
    kind = str(payload.get("kind", "") or "")
    agent = str(payload.get("agent", "") or "")
    tool = str(payload.get("tool", "") or "")
    label = str(payload.get("label", "") or _EXPERT_LABELS.get(agent, agent or "Agent"))
    status = str(payload.get("status", "") or "")
    detail = str(payload.get("detail", "") or payload.get("reason", "") or payload.get("summary", "") or "")
    input_value = payload.get("input")

    if kind == "expert_start":
        detail = str(payload.get("task", "") or "模块开始执行。")
        title = f"{_EXPERT_LABELS.get(agent, agent)}开始"
    elif kind == "expert_done":
        title = f"{_EXPERT_LABELS.get(agent, agent)}完成"
        detail = f"已生成 {payload.get('output_count', 0)} 个专家输出。"
    elif kind == "expert_skipped":
        title = f"{_EXPERT_LABELS.get(agent, agent)}跳过"
    elif kind == "llm_call":
        title = f"{label}调用模型"
    elif kind == "tool_start":
        title = f"{_EXPERT_LABELS.get(agent, agent)}调用工具"
        detail = f"{tool} 输入: {_short_json(input_value)}"
    elif kind == "tool_done":
        result_text = "成功" if status == "success" else "失败"
        title = f"{tool} {result_text}"
    elif kind == "expert_detail":
        title = f"{_EXPERT_LABELS.get(agent, agent)}状态"
    else:
        title = label or kind or "Agent 状态"

    return {
        "kind": kind,
        "agent": agent,
        "label": label,
        "title": title,
        "detail": detail[:500],
        "tool": tool,
        "status": status,
    }


def _short_json(value: Any, limit: int = 220) -> str:
    if value is None:
        return "{}"
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


class MultiAgentService:
    """Service layer that exposes the multi-agent graph as a simple chat API.

    Usage::

        service = MultiAgentService()
        result = service.chat(
            message="总结这篇关于 SCMA 的论文",
            history=[],
            temperature=0.3,
            max_steps=12,
        )
        print(result["answer"])
        for step in result["agent_trace"]:
            print(f"  -> {step['agent']}: {step['summary']}")
    """

    def __init__(self):
        self._graph = get_graph()
        self._memory_store = get_agent_memory_store() if conf.ENABLE_AGENT_MEMORY else None

    # ---- initial state helper ------------------------------------------------

    def _build_initial_state(
        self,
        message: str,
        history: list[Dict[str, str]] | None = None,
        session_id: str | None = None,
        code_workspace_path: str | None = None,
        code_workspace_is_project: bool = False,
        code_python_executable: str | None = None,
    ) -> MultiAgentState:
        """Build the initial graph state from user message and chat history."""
        history = history or []
        normalized_session_id = AgentMemoryStore.normalize_session_id(
            session_id or f"session_{uuid.uuid4().hex[:8]}"
        )
        memory = (
            self._memory_store.load(normalized_session_id)
            if self._memory_store is not None
            else {"session_id": normalized_session_id}
        )
        messages: list = []
        for item in history[-8:]:
            role = item.get("role", "")
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                if role == "user":
                    messages.append(HumanMessage(content=content[:2000]))
                else:
                    messages.append(HumanMessage(content=content[:2000], name="assistant"))
        messages.append(HumanMessage(content=message.strip()))

        return {
            "messages": messages,
            "session_id": normalized_session_id,
            "memory": memory,
            "next_agent": "supervisor",
            "expert_outputs": [],
            "task_plan": [],
            "module_tasks": {},
            "current_task": message.strip(),
            "active_skills": list(memory.get("active_skills", [])) if isinstance(memory, dict) else [],
            "skill_overrides": {},
            "code_workspace_path": str(code_workspace_path or conf.CODE_BUILDER_WORKSPACE_DIR).strip(),
            "code_workspace_is_project": bool(code_workspace_is_project),
            "code_python_executable": str(code_python_executable or conf.CODE_BUILDER_PYTHON_EXECUTABLE).strip(),
        }

    def clear_memory(self, session_id: str) -> Dict[str, Any]:
        """Clear persisted memory for one session."""
        normalized_session_id = AgentMemoryStore.normalize_session_id(session_id)
        if self._memory_store is not None:
            self._memory_store.clear(normalized_session_id)
        return {"status": "ok", "session_id": normalized_session_id}

    def get_memory(self, session_id: str) -> Dict[str, Any]:
        """Return persisted memory for one session."""
        normalized_session_id = AgentMemoryStore.normalize_session_id(session_id)
        if self._memory_store is None:
            return {"session_id": normalized_session_id, "memory_enabled": False}
        memory = self._memory_store.load(normalized_session_id)
        memory["memory_enabled"] = True
        return memory

    def _persist_memory_after_run(
        self,
        session_id: str,
        message: str,
        final_answer: str,
        expert_outputs: List[Dict[str, Any]],
        active_skills: List[str],
    ) -> Dict[str, Any]:
        """Persist memory if enabled and return the updated snapshot."""
        if self._memory_store is None:
            return {"session_id": session_id}
        return self._memory_store.update_after_run(
            session_id=session_id,
            user_message=message,
            final_answer=final_answer,
            expert_outputs=expert_outputs,
            active_skills=active_skills,
        )

    # ---- streaming API -------------------------------------------------------

    def chat_stream(
        self,
        message: str,
        history: list[Dict[str, str]] | None = None,
        temperature: float = 0.1,
        max_steps: int = 20,
        session_id: str | None = None,
        code_workspace_path: str | None = None,
        code_workspace_is_project: bool = False,
        code_python_executable: str | None = None,
    ) -> Generator[str, None, None]:
        """Run the multi-agent workflow and yield SSE events for live progress.

        Yields Server-Sent Event strings that the frontend can consume
        via ``EventSource`` or ``fetch()`` with a ``ReadableStream``.
        """
        initial_state = self._build_initial_state(
            message,
            history,
            session_id=session_id,
            code_workspace_path=code_workspace_path,
            code_workspace_is_project=code_workspace_is_project,
            code_python_executable=code_python_executable,
        )
        session_id = initial_state["session_id"]
        thread_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(12, int(max_steps) * 2),
        }

        final_answer: Optional[str] = None
        step_count = 0
        expert_outputs_for_memory: List[Dict[str, Any]] = []
        active_skills_for_memory: List[str] = list(initial_state.get("active_skills", []))

        logger.info(
            f"[multi-agent] stream start session={session_id} thread={thread_id} max_steps={max_steps}"
        )
        yield _sse_event(
            "status",
            {
                "status": "started",
                "session_id": session_id,
                "thread_id": thread_id,
                "memory_turns": initial_state.get("memory", {}).get("turn_count", 0),
                "code_workspace_path": initial_state.get("code_workspace_path", ""),
                "code_workspace_is_project": initial_state.get("code_workspace_is_project", False),
                "code_python_executable": initial_state.get("code_python_executable", ""),
            },
        )

        event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def progress_sink(event: str, data: dict) -> None:
            event_queue.put(("progress", {"kind": event, **data}))

        def run_graph() -> None:
            token = set_progress_sink(progress_sink)
            try:
                for graph_step in self._graph.stream(initial_state, config, stream_mode="updates"):
                    event_queue.put(("graph_step", graph_step))
            except Exception as exc:
                event_queue.put(("exception", exc))
            finally:
                reset_progress_sink(token)
                event_queue.put(("graph_done", None))

        worker = threading.Thread(target=run_graph, name=f"agent-stream-{thread_id}", daemon=True)
        worker.start()

        while True:
            try:
                event_type, payload = event_queue.get(timeout=0.5)
            except queue.Empty:
                if shutdown_requested():
                    logger.info("[multi-agent] stream cancelled by shutdown request")
                    yield _sse_event("error", {"message": "服务正在关闭，当前工作流已中止。"})
                    return
                continue

            if event_type == "progress":
                yield _sse_event("progress", _normalize_progress_payload(payload))
                continue

            if event_type == "exception":
                logger.info(f"[multi-agent] stream failed: {payload}")
                yield _sse_event("error", {"message": f"多 Agent 工作流出错: {payload}"})
                return

            if event_type == "graph_done":
                break

            step_output = payload
            step_count += 1
            for node_name, node_update in step_output.items():
                if shutdown_requested():
                    logger.info("[multi-agent] stream cancelled during step by shutdown request")
                    yield _sse_event("error", {"message": "服务正在关闭，当前工作流已中止。"})
                    return
                if node_name == "supervisor":
                    na = node_update.get("next_agent", "")
                    reason = ""
                    ct = node_update.get("current_task", "")
                    active_skills_for_memory = list(
                        node_update.get("active_skills", active_skills_for_memory)
                    )
                    # Extract reason from messages
                    for m in node_update.get("messages", []):
                        if isinstance(m, HumanMessage) and getattr(m, "name", None) == "supervisor":
                            reason = _strip_supervisor_control_prefix(str(m.content))[:200]

                    if na == "FINISH":
                        yield _sse_event("supervisor", {
                            "decision": "FINISH",
                            "label": "正在生成最终回复...",
                        })
                    elif na == "module_executor":
                        yield _sse_event("supervisor", {
                            "decision": "module_executor",
                            "label": "统一调度模块",
                            "task": ct[:200],
                            "reason": reason,
                        })
                    elif na in EXPERT_NAMES:
                        yield _sse_event("supervisor", {
                            "decision": na,
                            "label": _EXPERT_LABELS.get(na, na),
                            "task": ct[:200],
                            "reason": reason,
                        })

                elif node_name == "module_executor":
                    eos = node_update.get("expert_outputs", [])
                    expert_outputs_for_memory.extend(eos)
                    for eo in eos:
                        agent = eo.get("expert_name", "unknown")
                        content = eo.get("content", "")
                        yield _sse_event("expert_output", {
                            "agent": agent,
                            "label": _EXPERT_LABELS.get(agent, agent),
                            "summary": content[:300].replace("\n", " "),
                            "content_preview": content[:2000],
                        })

                elif node_name in EXPERT_NAMES:
                    eos = node_update.get("expert_outputs", [])
                    expert_outputs_for_memory.extend(eos)
                    for eo in eos:
                        content = eo.get("content", "")
                        yield _sse_event("expert_output", {
                            "agent": node_name,
                            "label": _EXPERT_LABELS.get(node_name, node_name),
                            "summary": content[:300].replace("\n", " "),
                            "content_preview": content[:2000],
                        })

                # Capture final synthesis
                for m in node_update.get("messages", []):
                    if _is_final_supervisor_message(m):
                        final_answer = str(m.content).strip()

        answer = final_answer or "多 Agent 工作流已完成，但未生成最终回复。"
        updated_memory = self._persist_memory_after_run(
            session_id=session_id,
            message=message,
            final_answer=answer,
            expert_outputs=expert_outputs_for_memory,
            active_skills=active_skills_for_memory,
        )
        yield _sse_event(
            "done",
            {
                "answer": answer,
                "steps": step_count,
                "session_id": session_id,
                "memory_turns": updated_memory.get("turn_count", 0),
            },
        )
        logger.info(
            f"[multi-agent] stream done session={session_id} steps={step_count} answer_len={len(answer)}"
        )

    # ---- synchronous API -----------------------------------------------------

    def chat(
        self,
        message: str,
        history: list[Dict[str, str]] | None = None,
        temperature: float = 0.1,
        max_steps: int = 20,
        session_id: str | None = None,
        code_workspace_path: str | None = None,
        code_workspace_is_project: bool = False,
        code_python_executable: str | None = None,
    ) -> Dict[str, Any]:
        """Run the multi-agent workflow for a user message (blocking)."""
        initial_state = self._build_initial_state(
            message,
            history,
            session_id=session_id,
            code_workspace_path=code_workspace_path,
            code_workspace_is_project=code_workspace_is_project,
            code_python_executable=code_python_executable,
        )

        session_id = initial_state["session_id"]
        thread_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(12, int(max_steps) * 2),
        }

        agent_trace: list[Dict[str, Any]] = []
        final_answer = ""
        status = "success"
        expert_outputs_for_memory: List[Dict[str, Any]] = []
        active_skills_for_memory: List[str] = list(initial_state.get("active_skills", []))

        try:
            logger.info(f"[multi-agent] start session={session_id} thread={thread_id} max_steps={max_steps}")
            for step_output in self._graph.stream(
                initial_state, config, stream_mode="updates"
            ):
                for node_name, node_update in step_output.items():
                    if node_name == "supervisor":
                        na = node_update.get("next_agent", "")
                        active_skills_for_memory = list(
                            node_update.get("active_skills", active_skills_for_memory)
                        )
                        if na == "FINISH":
                            agent_trace.append({"agent": "supervisor", "summary": "FINISH — 生成最终回复"})
                        elif na == "module_executor":
                            agent_trace.append({"agent": "supervisor", "summary": "统一调度模块"})
                        elif na:
                            agent_trace.append({"agent": "supervisor", "summary": f"→ {na}"})
                    elif node_name == "module_executor":
                        eos = node_update.get("expert_outputs", [])
                        expert_outputs_for_memory.extend(eos)
                        for eo in eos:
                            agent = eo.get("expert_name", "unknown")
                            summary = eo.get("content", "")[:200].replace("\n", " ")
                            agent_trace.append({"agent": agent, "summary": f"{summary}..."})
                    elif node_name in EXPERT_NAMES:
                        eos = node_update.get("expert_outputs", [])
                        expert_outputs_for_memory.extend(eos)
                        for eo in eos:
                            summary = eo.get("content", "")[:200].replace("\n", " ")
                            agent_trace.append({"agent": node_name, "summary": f"{summary}..."})

                    msgs = node_update.get("messages", [])
                    for m in msgs:
                        if _is_final_supervisor_message(m):
                            final_answer = str(m.content).strip()

        except Exception as exc:
            logger.info(f"[multi-agent] graph execution failed: {exc}")
            status = "fail"
            final_answer = f"多 Agent 工作流出错: {exc}"

        if not final_answer:
            final_answer = "多 Agent 工作流已完成，但未生成最终回复。请查看 agent_trace 了解详情。"

        updated_memory = self._persist_memory_after_run(
            session_id=session_id,
            message=message,
            final_answer=final_answer,
            expert_outputs=expert_outputs_for_memory,
            active_skills=active_skills_for_memory,
        )

        logger.info(
            f"[multi-agent] done session={session_id} status={status} "
            f"answer_len={len(final_answer)} steps={len(agent_trace)}"
        )
        return {
            "status": status,
            "answer": final_answer,
            "agent_trace": agent_trace,
            "session_id": session_id,
            "memory": updated_memory,
            "code_workspace_path": initial_state.get("code_workspace_path", ""),
            "code_workspace_is_project": initial_state.get("code_workspace_is_project", False),
            "code_python_executable": initial_state.get("code_python_executable", ""),
        }
