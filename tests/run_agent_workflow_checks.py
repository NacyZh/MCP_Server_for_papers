"""No-pytest regression checks for the multi-agent workflow.

Run with:
    python tests/run_agent_workflow_checks.py

These checks avoid real LLM, database, and network calls by exercising
pure helpers and monkeypatching supervisor dependencies where needed.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.messages import HumanMessage

from scholar_agent.agents import executor as executor_mod
from scholar_agent.agents.experts import code_builder as code_builder_mod
from scholar_agent.agents.experts import database_manager as database_manager_mod
from scholar_agent.agents.experts import literature as literature_mod
from scholar_agent.agents import supervisor as supervisor_mod
from scholar_agent.agents import graph as graph_mod
from scholar_agent.agents import synthesis as synthesis_mod
from scholar_agent.agents.graph import _route_supervisor
from scholar_agent.agents.service import _is_final_supervisor_message
from scholar_agent.agents.service import MultiAgentService
from scholar_agent.agents.state import append_expert_outputs
from scholar_agent.agents.planning import infer_task_plan, normalize_task_plan
from scholar_agent.agents.supervisor import supervisor_node
from scholar_agent.agents.synthesis import synthesis_node
from scholar_agent.config import conf
from scholar_agent.agents.utils import (
    collect_paper_ids_from_outputs,
    collect_retrieval_query_from_outputs,
    extract_user_query,
    filter_chunks_by_paper_ids,
    no_papers_response,
)
from scholar_agent.prompts import CODE_BUILDER_SYSTEM_PROMPT
from scholar_agent.core.logging import configure_logging, get_logger
from scholar_agent.core.runtime import clear_shutdown_request, request_shutdown
from scholar_agent.storage.memory_store import AgentMemoryStore
from scholar_agent.tools import code_tools as code_tools_mod
from scholar_agent.tools import writing_tools as writing_tools_mod
from scholar_agent.tools.paper_tools import ArxivSearchTool, LocalSearchTool
from scholar_agent.tools.registry import get_tool_registry


@contextmanager
def _patched(module: Any, name: str, replacement: Any):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def _state(
    query: str = "请根据论文方法生成 MATLAB 复现代码",
    expert_outputs: list[dict] | None = None,
    task_plan: list[str] | None = None,
) -> dict:
    return {
        "messages": [HumanMessage(content="旧问题"), HumanMessage(content=query)],
        "session_id": "test-session",
        "memory": {
            "session_id": "test-session",
            "summary": "",
            "user_preferences": [],
            "recent_topics": [],
            "recent_paper_ids": [],
            "active_skills": [],
            "turn_count": 0,
        },
        "next_agent": "supervisor",
        "expert_outputs": expert_outputs or [],
        "task_plan": task_plan or [],
        "module_tasks": {},
        "current_task": query,
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": code_tools_mod.CODE_WORKSPACE_DIR,
        "code_workspace_is_project": False,
        "code_python_executable": conf.CODE_BUILDER_PYTHON_EXECUTABLE,
    }


def _run_check(name: str, fn: Callable[[], None]) -> None:
    fn()
    print(f"PASS {name}")


def check_task_plan_inference() -> None:
    assert infer_task_plan("请用 MATLAB 复现论文算法") == []
    assert infer_task_plan("请完成完整研究任务，从文献到代码") == []
    assert infer_task_plan("分析这篇论文的方法和公式") == []
    assert infer_task_plan("总结这篇论文") == []
    assert infer_task_plan("检索 SCMA 相关文献") == []
    assert infer_task_plan("请帮我看看") == []
    assert infer_task_plan("SCMA") == []


def check_default_workspace_layout() -> None:
    work_root = Path(conf.SCHOLAR_AGENT_WORK_ROOT)
    code_root = Path(conf.CODE_BUILDER_WORKSPACE_DIR)
    document_root = Path(conf.WRITING_WORKSPACE_DIR)
    assert work_root.as_posix().endswith("scholar agent")
    assert code_root.parent == work_root
    assert document_root.parent == work_root
    assert code_root.name == "scholar code"
    assert document_root.name == "scholar document"


def check_plan_normalization() -> None:
    assert normalize_task_plan(["Literature", "literature", "FINISH", "bad", "METHODology"]) == [
        "literature",
        "methodology",
    ]


def check_module_executor_runs_plan_in_order() -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_literature(state: dict) -> dict:
        calls.append(("literature", state["current_task"], len(state["expert_outputs"])))
        return {
            "expert_outputs": [{"expert_name": "literature", "content": "lit"}],
            "messages": [HumanMessage(content="lit", name="literature")],
        }

    def fake_summarizer(state: dict) -> dict:
        calls.append(("summarizer", state["current_task"], len(state["expert_outputs"])))
        return {
            "expert_outputs": [{"expert_name": "summarizer", "content": "sum"}],
            "messages": [HumanMessage(content="sum", name="summarizer")],
        }

    state = _state(query="模块执行测试", task_plan=["literature", "summarizer"])
    state["module_tasks"] = {
        "literature": "literature task",
        "summarizer": "summarizer task",
    }
    with _patched(
        executor_mod,
        "_EXPERT_NODES",
        {"literature": fake_literature, "summarizer": fake_summarizer},
    ):
        update = executor_mod.module_executor_node(state)

    assert calls == [
        ("literature", "literature task", 0),
        ("summarizer", "summarizer task", 1),
    ]
    assert update["next_agent"] == "synthesis"
    assert [item["expert_name"] for item in update["expert_outputs"]] == ["literature", "summarizer"]


def check_module_executor_skips_downstream_after_no_papers() -> None:
    calls: list[str] = []

    def fake_literature(state: dict) -> dict:
        calls.append("literature")
        return no_papers_response("literature")

    def fake_code_builder(state: dict) -> dict:
        calls.append("code_builder")
        return {"expert_outputs": [{"expert_name": "code_builder", "content": "bad"}]}

    state = _state(query="生成代码", task_plan=["literature", "code_builder"])
    with _patched(
        executor_mod,
        "_EXPERT_NODES",
        {"literature": fake_literature, "code_builder": fake_code_builder},
    ):
        update = executor_mod.module_executor_node(state)

    assert calls == ["literature"]
    assert [item["expert_name"] for item in update["expert_outputs"]] == ["literature"]


def check_expert_output_reducer() -> None:
    existing = [{"expert_name": "literature", "content": "a"}]
    updates = [{"expert_name": "summarizer", "content": "b"}]
    assert append_expert_outputs(existing, updates) == existing + updates
    assert append_expert_outputs(None, updates) == updates
    assert append_expert_outputs(existing, None) == existing


def check_paper_selection_helpers() -> None:
    state = _state(
        expert_outputs=[
            {"expert_name": "literature", "metadata": {"paper_ids": ["p1", "p2"]}},
            {"expert_name": "summarizer", "metadata": {"paper_ids": ["p2", "p3"]}},
        ]
    )
    assert collect_paper_ids_from_outputs(state, preferred_experts=["literature"]) == ["p1", "p2"]
    chunks = [
        {"paper_id": "p0", "content": "ignored"},
        {"paper_id": "p2", "content": "kept"},
    ]
    assert filter_chunks_by_paper_ids(chunks, ["p2"]) == [chunks[1]]
    assert filter_chunks_by_paper_ids(chunks, ["missing"]) == chunks
    query_state = _state(
        expert_outputs=[
            {
                "expert_name": "literature",
                "metadata": {
                    "retrieval_query": 'all:"scma" AND all:"detection"',
                    "local_query": "SCMA detection message passing deep learning receiver",
                },
            }
        ]
    )
    assert (
        collect_retrieval_query_from_outputs(query_state, preferred_experts=["literature"])
        == "SCMA detection message passing deep learning receiver"
    )


def check_graph_route_sanitization() -> None:
    assert _route_supervisor({"next_agent": "finish"}) == "FINISH"
    assert _route_supervisor({"next_agent": "Literature"}) == "FINISH"
    assert _route_supervisor({"next_agent": "Literature", "task_plan": ["literature"]}) == "module_executor"
    assert _route_supervisor({"next_agent": "module_executor", "task_plan": ["literature"]}) == "module_executor"
    assert _route_supervisor({"next_agent": "not-an-agent"}) == "FINISH"


def check_latest_user_query_and_no_papers_message() -> None:
    assert extract_user_query(_state(query="最新问题")) == "最新问题"
    response = no_papers_response("summarizer")
    assert response["expert_outputs"][0]["metadata"]["error"] == "no_papers_found"
    assert response["messages"][0].name == "summarizer"


def check_search_tool_schema_guides_llm_query_arguments() -> None:
    local_text = f"{LocalSearchTool.description} {LocalSearchTool.params['properties']['query']['description']}"
    arxiv_text = f"{ArxivSearchTool.description} {ArxivSearchTool.params['properties']['query']['description']}"

    assert "Do not pass" in local_text
    assert "planning text" in local_text
    assert "Concise academic search query" in local_text

    assert "Concise English search keywords" in arxiv_text
    assert "Translate the research topic to English" in arxiv_text
    assert "Do not pass Chinese task descriptions" in arxiv_text


def check_literature_tool_call_dispatch_uses_llm_arguments() -> None:
    class FakeToolResult:
        def __init__(self, status: str = "success", result: str = "ok", data: Any = None):
            self.status = status
            self.result = result
            self.data = data

    class FakeTool:
        def __init__(self, name: str):
            self.name = name
            self.calls: list[dict] = []

        def execute(self, **kwargs: Any) -> FakeToolResult:
            self.calls.append(kwargs)
            return FakeToolResult(data=[])

    class FakeRegistry:
        def __init__(self):
            self.tools = {
                "list_local_database": FakeTool("list_local_database"),
                "search_arxiv_papers": FakeTool("search_arxiv_papers"),
                "search_local_papers_chunks": FakeTool("search_local_papers_chunks"),
            }

        def get(self, name: str) -> FakeTool:
            return self.tools[name]

        def to_langchain_tool(self, name: str) -> FakeTool:
            return self.tools[name]

    class FakeBoundLLM:
        def invoke(self, messages: list) -> Any:
            return type(
                "FakeResponse",
                (),
                {
                    "tool_calls": [
                        {"name": "list_local_database", "args": {}},
                        {
                            "name": "search_arxiv_papers",
                            "args": {"query": "SCMA detection message passing", "max_results": 3},
                        },
                    ]
                },
            )()

    class FakeLLM:
        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return FakeBoundLLM()

    registry = FakeRegistry()
    task = "围绕用户问题检索本地数据库和 arXiv，筛选最相关论文: 请围绕 SCMA 完成完整研究任务"
    literature_mod._run_literature_tool_calls(FakeLLM(), registry, "请检索 SCMA 检测相关论文", task, _state())

    assert registry.tools["search_arxiv_papers"].calls == [
        {"query": "SCMA detection message passing", "max_results": 3}
    ]
    assert task not in str(registry.tools["search_arxiv_papers"].calls)


def check_database_manager_deletes_with_tool_result() -> None:
    class FakeToolResult:
        status = "success"
        result = "Deleted paper local_75ea7b80 from SQLite and vector store."
        data = None

    class FakeTool:
        name = "delete_paper_from_database"
        description = "Delete local paper"
        params = {
            "type": "object",
            "properties": {"local_id": {"type": "string"}},
            "required": ["local_id"],
            "additionalProperties": False,
        }

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def execute(self, **kwargs: Any) -> FakeToolResult:
            self.calls.append(kwargs)
            return FakeToolResult()

    class FakeRegistry:
        def __init__(self) -> None:
            self.tool = FakeTool()

        def get(self, name: str) -> FakeTool | None:
            return self.tool if name == "delete_paper_from_database" else None

        def to_langchain_tool(self, name: str) -> FakeTool:
            return self.tool

    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None) -> None:
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    tool_calls=[
                        {
                            "name": "delete_paper_from_database",
                            "args": {"local_id": "local_75ea7b80"},
                            "id": "delete-1",
                        }
                    ]
                )
            return FakeResponse("删除工具已返回成功。")

    class FakeLLM:
        def __init__(self) -> None:
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

    registry = FakeRegistry()
    with _patched(database_manager_mod, "get_tool_registry", lambda: registry):
        with _patched(database_manager_mod, "ChatOpenAI", lambda **kwargs: FakeLLM()):
            update = database_manager_mod.database_manager_node(
                _state(query="删除本地数据库中ID: local_75ea7b80的数据")
            )

    assert registry.tool.calls == [{"local_id": "local_75ea7b80"}]
    output = update["expert_outputs"][0]
    assert output["expert_name"] == "database_manager"
    assert output["metadata"]["tool_calls"][0]["tool"] == "delete_paper_from_database"
    assert output["metadata"]["tool_calls"][0]["status"] == "success"
    assert "Deleted paper local_75ea7b80" in output["content"]


def check_writing_document_tools_docx_and_tex() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_workspace = writing_tools_mod.conf.WRITING_WORKSPACE_DIR
        writing_tools_mod.conf.WRITING_WORKSPACE_DIR = tmpdir
        try:
            docx_result = writing_tools_mod.WritingWriteDocxDocumentTool().execute(
                path="draft/polished.docx",
                title="Polished Draft",
                content="# Abstract\n\nThis paper presents a concise academic contribution.\n\n- Clear contribution",
            )
            assert docx_result.status == "success"
            docx_text = writing_tools_mod.WritingReadDocxDocumentTool().execute(path="draft/polished.docx")
            assert docx_text.status == "success"
            assert "Polished Draft" in docx_text.result
            assert "This paper presents" in docx_text.result

            tex_result = writing_tools_mod.WritingWriteLatexDocumentTool().execute(
                path="draft/main.tex",
                content="\\section{Introduction}\nThis paper improves clarity.",
            )
            assert tex_result.status == "success"
            tex_text = writing_tools_mod.WritingReadLatexDocumentTool().execute(path="draft/main.tex")
            assert tex_text.status == "success"
            assert "\\section{Introduction}" in tex_text.result

            compile_result = writing_tools_mod.WritingCompileLatexDocumentTool().execute(
                path="draft/main.tex",
                engine="auto",
                timeout_s=10,
            )
            assert compile_result.status in {"success", "fail"}
            if compile_result.status == "success":
                assert "PDF:" in compile_result.result
            else:
                assert "LaTeX" in compile_result.result

            files = writing_tools_mod.WritingListDocumentsTool().execute()
            assert "draft/polished.docx" in files.result.replace("\\", "/")
            assert "draft/main.tex" in files.result.replace("\\", "/")
        finally:
            writing_tools_mod.conf.WRITING_WORKSPACE_DIR = original_workspace


def check_synthesis_does_not_claim_missing_writing_documents() -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("synthesis LLM should not run for failed writing delivery")

    state = _state(query="总结SCMA论文并生成中文Word和英文LaTeX")
    state["expert_outputs"] = [
        {"expert_name": "summarizer", "content": "SCMA 论文摘要内容", "metadata": {"paper_ids": ["p1"]}},
        {
            "expert_name": "writing_editor",
            "content": "论文写作文档生成未完成：本轮没有成功的文档写入工具调用。",
            "metadata": {
                "generated_documents": [],
                "tool_calls": [
                    {
                        "tool": "writing_read_document",
                        "args": {"path": "writing_output/summary.md"},
                        "status": "fail",
                        "result": "Document not found",
                    }
                ],
                "writing_workspace": "workspace/writing",
            },
        },
    ]

    with _patched(synthesis_mod, "ChatOpenAI", FakeLLM):
        update = synthesis_node(state)

    answer = update["messages"][0].content
    assert "文档生成未完成" in answer
    assert "没有任何 `writing_write_docx_document`" in answer
    assert "已生成" not in answer
    assert "writing_output/SCMA_Literature_Summary_中文.docx" not in answer


def check_code_builder_quality_review_revision() -> None:
    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse("初版代码")
            if self.calls == 2:
                return FakeResponse("## 质量结论\nNEEDS_REVISION\n\n## 必须修改\n- 补入口脚本")
            return FakeResponse("修订后完整代码")

    fake_llm = FakeLLM()

    def fake_chat_openai(**kwargs: Any) -> FakeLLM:
        return fake_llm

    with tempfile.TemporaryDirectory() as tmpdir:
        original_workspace = code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR
        code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR = tmpdir
        try:
            state = _state(
                query="请生成代码",
                expert_outputs=[{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
            )
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
        finally:
            code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR = original_workspace

    output = update["expert_outputs"][0]
    assert output["content"] == "修订后完整代码"
    assert output["metadata"]["quality_reviewed"] is True
    assert output["metadata"]["revised_after_review"] is True


def check_code_builder_prompt_matches_autonomous_project_workflow() -> None:
    assert "项目级自主编码代理" in CODE_BUILDER_SYSTEM_PROMPT
    assert "code_workspace_set_validation_plan" in CODE_BUILDER_SYSTEM_PROMPT
    assert "code_workspace_record_validation" in CODE_BUILDER_SYSTEM_PROMPT
    assert "最后一次文件修改后" in CODE_BUILDER_SYSTEM_PROMPT
    assert "不要直接输出大段代码当作完成" in CODE_BUILDER_SYSTEM_PROMPT


def check_code_workspace_tools_write_and_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original = code_tools_mod.CODE_WORKSPACE_DIR
        original_python = code_tools_mod.conf.CODE_BUILDER_PYTHON_EXECUTABLE
        code_tools_mod.CODE_WORKSPACE_DIR = tmpdir
        code_tools_mod.conf.CODE_BUILDER_PYTHON_EXECUTABLE = sys.executable
        try:
            registry = get_tool_registry()
            assert registry.get("code_workspace_write_file").execute(
                path="demo/main.py",
                content="print('ok')\n",
            ).status == "success"
            run_result = registry.get("code_workspace_run_python").execute(path="demo/main.py", timeout_s=5)
            assert run_result.status == "success"
            assert "ok" in run_result.result
            shell_result = registry.get("code_workspace_run_shell").execute(
                command="python -c \"print('shell ok')\"",
                cwd="demo",
                timeout_s=10,
            )
            assert shell_result.status == "success"
            assert "shell ok" in shell_result.result
            shell_python_result = registry.get("code_workspace_run_shell").execute(
                command="python -c \"import sys; print(sys.executable)\"",
                cwd="demo",
                timeout_s=10,
            )
            assert shell_python_result.status == "success"
            assert Path(sys.executable).resolve() == Path(shell_python_result.result.strip().splitlines()[-1]).resolve()
            with code_tools_mod.use_code_python_executable(sys.executable) as selected_python:
                assert Path(selected_python).resolve() == Path(sys.executable).resolve()
                contextual_run = registry.get("code_workspace_run_shell").execute(
                    command="python -c \"import sys; print(sys.executable)\"",
                    cwd="demo",
                    timeout_s=10,
                )
            assert contextual_run.status == "success"
            assert Path(contextual_run.result.strip().splitlines()[-1]).resolve() == Path(sys.executable).resolve()
            blocked_shell = registry.get("code_workspace_run_shell").execute(command="Remove-Item -Recurse -Force .")
            assert blocked_shell.status == "fail"
            patch_result = registry.get("code_workspace_make_patch").execute(
                path="demo/main.py",
                new_content="print('patched')\n",
            )
            assert patch_result.status == "success"
            assert "--- a/demo/main.py" in patch_result.result
            apply_result = registry.get("code_workspace_apply_patch").execute(patch=patch_result.result)
            assert apply_result.status == "success"
            assert (Path(tmpdir) / "demo" / "main.py").read_text(encoding="utf-8") == "print('patched')\n"
            escape_result = registry.get("code_workspace_write_file").execute(path="../bad.py", content="")
            assert escape_result.status == "fail"
            bad_patch = "--- a/../bad.py\n+++ b/../bad.py\n@@ -0,0 +1 @@\n+x\n"
            assert registry.get("code_workspace_apply_patch").execute(patch=bad_patch).status == "fail"
        finally:
            code_tools_mod.CODE_WORKSPACE_DIR = original
            code_tools_mod.conf.CODE_BUILDER_PYTHON_EXECUTABLE = original_python


def check_service_initial_state_accepts_code_python_executable() -> None:
    service = MultiAgentService()
    with tempfile.TemporaryDirectory() as tmpdir:
        state = service._build_initial_state(
            "生成代码",
            session_id="check-runtime",
            code_workspace_path=tmpdir,
            code_workspace_is_project=True,
            code_python_executable=sys.executable,
        )
        assert state["code_workspace_path"] == tmpdir
        assert state["code_workspace_is_project"] is True
        assert Path(state["code_python_executable"]).resolve() == Path(sys.executable).resolve()


def check_code_builder_includes_all_external_mcp_tools() -> None:
    class FakeTool:
        def __init__(self, is_external: bool = False):
            self.is_external_mcp_tool = is_external

    class FakeRegistry:
        def __init__(self):
            self.tools = {
                "code_workspace_write_file": FakeTool(),
                "code_workspace_run_shell": FakeTool(),
                "github__create_issue": FakeTool(is_external=True),
                "jupyter__run_cell": FakeTool(is_external=True),
                "search_local_papers_chunks": FakeTool(is_external=False),
            }

        def list_all(self) -> list[str]:
            return sorted(self.tools)

        def get(self, name: str) -> FakeTool | None:
            return self.tools.get(name)

    names = code_builder_mod._available_code_builder_tool_names(FakeRegistry())
    assert "code_workspace_write_file" in names
    assert "code_workspace_run_shell" in names
    assert "github__create_issue" in names
    assert "jupyter__run_cell" in names
    assert "search_local_papers_chunks" not in names


def check_code_builder_autonomous_loop_uses_tools() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "code_workspace_write_file",
                            "args": {"path": "demo/main.py", "content": "print('ok')\n"},
                        }
                    ]
                )
            if self.calls == 2:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_patch",
                            "name": "code_workspace_make_patch",
                            "args": {"path": "demo/main.py", "new_content": "print('patched')\n"},
                        }
                    ]
                )
            if self.calls == 3:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_apply",
                            "name": "code_workspace_apply_patch",
                            "args": {
                                "patch": (
                                    "--- a/demo/main.py\n"
                                    "+++ b/demo/main.py\n"
                                    "@@ -1 +1 @@\n"
                                    "-print('ok')\n"
                                    "+print('patched')\n"
                                )
                            },
                        }
                    ]
                )
            if self.calls == 4:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_plan",
                            "name": "code_workspace_set_validation_plan",
                            "args": {"targets": "demo/main.py", "rationale": "smoke script must run"},
                        }
                    ]
                )
            if self.calls == 5:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_run",
                            "name": "code_workspace_run_python",
                            "args": {"path": "demo/main.py", "timeout_s": 5},
                        }
                    ]
                )
            return FakeResponse("最终说明: demo/main.py 已通过运行。")

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()
            self.plain_calls = 0

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            self.plain_calls += 1
            return FakeResponse("## 质量结论\nPASS")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = code_tools_mod.CODE_WORKSPACE_DIR
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_tools_mod.CODE_WORKSPACE_DIR = tmpdir
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(
                query="请生成代码",
                expert_outputs=[
                    {"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}
                ],
            )
            state["code_workspace_path"] = tmpdir
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert "最终说明" in output["content"]
            assert output["metadata"]["autonomous_coding"] is True
            assert "code_workspace_apply_patch" in output["metadata"]["tool_trace"]
            assert "code_workspace_set_validation_plan" in output["metadata"]["tool_trace"]
            assert "code_workspace_run_python" in output["metadata"]["tool_trace"]
            project_path = Path(output["metadata"]["code_project_path"])
            assert project_path == Path(tmpdir).resolve()
            assert (project_path / "demo" / "main.py").exists()
        finally:
            code_tools_mod.CODE_WORKSPACE_DIR = original_dir
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_code_builder_defers_final_until_files_and_validation() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse("过早结束")
            if self.calls == 2:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "code_workspace_write_file",
                            "args": {"path": "main.py", "content": "print('ok')\n"},
                        }
                    ]
                )
            if self.calls == 3:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_plan",
                            "name": "code_workspace_set_validation_plan",
                            "args": {"targets": "main.py", "rationale": "main entry must run"},
                        }
                    ]
                )
            if self.calls == 4:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_run",
                            "name": "code_workspace_run_python",
                            "args": {"path": "main.py", "timeout_s": 5},
                        }
                    ]
                )
            return FakeResponse("最终说明: 已创建并验证。")

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            return FakeResponse("## 质量结论\nPASS")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(
                query="请生成代码",
                expert_outputs=[
                    {"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}
                ],
            )
            state["code_workspace_path"] = tmpdir
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert output["content"] == "最终说明: 已创建并验证。"
            assert fake_llm.bound.calls == 5
            assert "code_workspace_write_file" in output["metadata"]["tool_trace"]
            assert "code_workspace_set_validation_plan" in output["metadata"]["tool_trace"]
            assert "code_workspace_run_python" in output["metadata"]["tool_trace"]
            assert "code_workspace_record_validation" not in output["metadata"]["tool_trace"]
        finally:
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_code_builder_requires_all_validation_plan_targets() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "code_workspace_write_file",
                            "args": {"path": "test_components.m", "content": "disp('components ok')\n"},
                        }
                    ]
                )
            if self.calls == 2:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_plan",
                            "name": "code_workspace_set_validation_plan",
                            "args": {
                                "targets": "test_components.m\nrun_simulation.m",
                                "rationale": "Both MATLAB validation scripts must run.",
                            },
                        }
                    ]
                )
            if self.calls == 3:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_record_one",
                            "name": "code_workspace_record_validation",
                            "args": {
                                "passed": True,
                                "tool": "matlab__run_matlab_file",
                                "target": "test_components.m",
                                "evidence": "components ok",
                            },
                        }
                    ]
                )
            if self.calls == 4:
                return FakeResponse("过早声称完成")
            if self.calls == 5:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_record_two",
                            "name": "code_workspace_record_validation",
                            "args": {
                                "passed": True,
                                "tool": "matlab__run_matlab_file",
                                "target": "run_simulation.m",
                                "evidence": "simulation ok",
                            },
                        }
                    ]
                )
            return FakeResponse("最终说明: 两个 MATLAB 脚本均已通过。")

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            return FakeResponse("## 质量结论\nPASS")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(
                query="运行 test_components.m 和 run_simulation.m 直到无错",
                expert_outputs=[
                    {"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}
                ],
            )
            state["code_workspace_path"] = tmpdir
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert output["content"] == "最终说明: 两个 MATLAB 脚本均已通过。"
            assert fake_llm.bound.calls == 6
            assert "code_workspace_set_validation_plan" in output["metadata"]["tool_trace"]
            assert output["metadata"]["tool_trace"].count("code_workspace_record_validation") == 2
        finally:
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_code_builder_continues_recent_incomplete_project() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "code_workspace_write_file",
                            "args": {"path": "main.py", "content": "print('ok')\n"},
                        }
                    ]
                )
            if self.calls == 2:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_plan",
                            "name": "code_workspace_set_validation_plan",
                            "args": {"targets": "main.py", "rationale": "entry must run"},
                        }
                    ]
                )
            if self.calls == 3:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_run",
                            "name": "code_workspace_run_python",
                            "args": {"path": "main.py", "timeout_s": 5},
                        }
                    ]
                )
            return FakeResponse("最终说明")

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            return FakeResponse("## 质量结论\nPASS")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_default = code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR = tmpdir
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        recent_project = Path(tmpdir) / "paper_local_123"
        recent_project.mkdir()
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(query="继续运行并修复验证脚本")
            state["expert_outputs"] = []
            state["code_workspace_path"] = tmpdir
            state["memory"]["recent_code_project_path"] = str(recent_project)
            state["memory"]["recent_code_delivery_status"] = "incomplete"
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert Path(output["metadata"]["code_project_path"]) == recent_project.resolve()
            assert output["metadata"]["code_project_slug"] == recent_project.name
            assert (recent_project / "main.py").exists()
        finally:
            code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR = original_default
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_code_builder_cancels_unvalidated_project_on_shutdown() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_write_readme",
                            "name": "code_workspace_write_file",
                            "args": {"path": "README.md", "content": "partial\n"},
                        }
                    ]
                )
            request_shutdown("test shutdown")
            return FakeResponse("声称完整但没有验证")

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()
            self.plain_calls = 0

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            self.plain_calls += 1
            return FakeResponse("不应调用质量审查")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(
                query="请生成代码",
                expert_outputs=[
                    {"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}
                ],
            )
            state["code_workspace_path"] = tmpdir
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert output["metadata"]["delivery_status"] == "cancelled"
            assert "代码构建已中止" in output["content"]
            assert fake_llm.plain_calls == 0
        finally:
            clear_shutdown_request()
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_code_builder_has_no_tool_round_budget() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 95:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_plan",
                            "name": "code_workspace_set_validation_plan",
                            "args": {"targets": "README.md", "rationale": "README exists"},
                        }
                    ]
                )
            if self.calls == 96:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_record",
                            "name": "code_workspace_record_validation",
                            "args": {
                                "passed": True,
                                "tool": "manual",
                                "target": "README.md",
                                "evidence": "README created after many edits",
                            },
                        }
                    ]
                )
            if self.calls == 97:
                return FakeResponse("最终说明: 已完成长轮次调试。")
            return FakeResponse(
                tool_calls=[
                    {
                        "id": f"call_write_{self.calls}",
                        "name": "code_workspace_write_file",
                        "args": {"path": "README.md", "content": f"partial {self.calls}\n"},
                    }
                ]
            )

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            return FakeResponse("unused")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(
                query="请生成代码",
                expert_outputs=[
                    {"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}
                ],
            )
            state["code_workspace_path"] = tmpdir
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert fake_llm.bound.calls == 97
            assert output["metadata"]["delivery_status"] == "complete"
            assert output["content"] == "最终说明: 已完成长轮次调试。"
        finally:
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_code_builder_ignores_unexpected_tool_arguments() -> None:
    class FakeResponse:
        def __init__(self, content: str = "", tool_calls: list | None = None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages: list) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "code_workspace_write_file",
                            "args": {"path": "main.py", "content": "print('ok')\n"},
                        }
                    ]
                )
            if self.calls == 2:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_read",
                            "name": "code_workspace_read_file",
                            "args": {"path": "main.py", "content": "unexpected stale field"},
                        }
                    ]
                )
            if self.calls == 3:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_plan",
                            "name": "code_workspace_set_validation_plan",
                            "args": {"targets": "main.py", "rationale": "entry must run"},
                        }
                    ]
                )
            if self.calls == 4:
                return FakeResponse(
                    tool_calls=[
                        {
                            "id": "call_run",
                            "name": "code_workspace_run_python",
                            "args": {"path": "main.py", "timeout_s": 5},
                        }
                    ]
                )
            return FakeResponse("最终说明: 已创建并验证。")

    class FakeLLM:
        def __init__(self):
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools: list) -> FakeBoundLLM:
            return self.bound

        def invoke(self, messages: list) -> FakeResponse:
            return FakeResponse("## 质量结论\nPASS")

    with tempfile.TemporaryDirectory() as tmpdir:
        original_autonomous = code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS
        code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = True
        fake_llm = FakeLLM()

        def fake_chat_openai(**kwargs: Any) -> FakeLLM:
            return fake_llm

        try:
            state = _state(
                query="请生成代码",
                expert_outputs=[
                    {"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}
                ],
            )
            state["code_workspace_path"] = tmpdir
            with _patched(code_builder_mod, "ChatOpenAI", fake_chat_openai):
                update = code_builder_mod.code_builder_node(state)
            output = update["expert_outputs"][0]
            assert output["content"] == "最终说明: 已创建并验证。"
            assert "Ignored unexpected argument(s): content" in output["metadata"]["tool_trace"]
        finally:
            code_builder_mod.conf.AGENT_CODE_BUILDER_AUTONOMOUS = original_autonomous


def check_memory_store_roundtrip_and_prompt() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentMemoryStore(str(Path(tmpdir) / "agent_memory.db"))
        session_id = AgentMemoryStore.normalize_session_id("web session / unsafe")
        assert session_id == "web_session_unsafe"

        before = store.load(session_id)
        assert before["turn_count"] == 0

        after = store.update_after_run(
            session_id=session_id,
            user_message="请用 MATLAB 复现这篇论文的方法",
            final_answer="已生成 MATLAB 代码，并引用 paper local_abc。",
            expert_outputs=[
                {
                    "expert_name": "methodology",
                    "content": "method",
                    "metadata": {"paper_ids": ["local_abc"]},
                },
                {
                    "expert_name": "code_builder",
                    "content": "code",
                    "metadata": {
                        "code_project_path": str(Path(tmpdir) / "demo_project"),
                        "code_project_slug": "demo_project",
                        "delivery_status": "complete",
                        "validation_evidence": "passed=True; tool=code_workspace_run_python; target=main.py",
                    },
                }
            ],
            active_skills=["matlab-expert"],
        )
        assert after["turn_count"] == 1
        assert after["user_preferences"] == []
        assert after["recent_paper_ids"] == ["local_abc"]
        assert after["active_skills"] == ["matlab-expert"]
        prompt_text = AgentMemoryStore.format_for_prompt(after)
        assert "local_abc" in prompt_text
        assert "最近代码项目路径" in prompt_text
        assert after["recent_code_delivery_status"] == "complete"

        store.clear(session_id)
        assert store.load(session_id)["turn_count"] == 0


def check_logging_writes_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "scholar_agent_test.log"
        configured = configure_logging(log_file=log_path, console=False, force=True)
        get_logger("test").info("workflow logging check")
        assert configured == log_path
        content = log_path.read_text(encoding="utf-8")
        assert "workflow logging check" in content
        configure_logging(log_file=ROOT_DIR / "workspace" / "logs" / "test_workflow.log", console=False, force=True)


def check_service_loads_and_updates_memory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentMemoryStore(str(Path(tmpdir) / "agent_memory.db"))
        service = object.__new__(MultiAgentService)
        service._graph = None
        service._memory_store = store

        store.update_after_run(
            session_id="service-memory",
            user_message="请用 Python 分析论文",
            final_answer="已按 Python 思路回答",
            expert_outputs=[],
            active_skills=[],
        )

        state = service._build_initial_state(
            "继续分析它的方法",
            history=[{"role": "user", "content": "上一轮问题"}, {"role": "assistant", "content": "上一轮回答"}],
            session_id="service-memory",
        )
        assert state["session_id"] == "service-memory"
        assert state["memory"]["turn_count"] == 1
        assert state["memory"]["user_preferences"] == []

        updated = service._persist_memory_after_run(
            session_id="service-memory",
            message="继续分析它的方法",
            final_answer="完成方法分析",
            expert_outputs=[{"expert_name": "methodology", "content": "x", "metadata": {"paper_ids": ["p1"]}}],
            active_skills=[],
        )
        assert updated["turn_count"] == 2
        assert updated["recent_paper_ids"][0] == "p1"


def check_final_answer_filter() -> None:
    routing = HumanMessage(content="Supervisor 调度: go literature", name="supervisor")
    module_plan = HumanMessage(content="Supervisor 模块计划: run modules", name="supervisor")
    final = HumanMessage(content="最终整合答案", name="supervisor")
    assert not _is_final_supervisor_message(routing)
    assert not _is_final_supervisor_message(module_plan)
    assert _is_final_supervisor_message(final)


def check_supervisor_direct_response_without_modules() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {
            "task_plan": [],
            "module_tasks": {},
            "reason": "LLM judged no academic module is needed",
            "skills": [],
            "direct_answer": "直接回复，不调用学术模块。",
        }

    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(_state(query="任意短输入"))

    assert update["next_agent"] == "FINISH"
    assert update["task_plan"] == []
    assert update["messages"][0].content == "直接回复，不调用学术模块。"


def check_supervisor_plans_modules_once_without_chain() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {
            "task_plan": ["literature", "summarizer", "methodology", "code_builder"],
            "module_tasks": {},
            "reason": "LLM selected all research modules",
            "skills": [],
            "direct_answer": "",
        }

    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(_state())

    assert update["next_agent"] == "module_executor"
    assert update["task_plan"] == ["literature", "summarizer", "methodology", "code_builder"]
    assert set(update["module_tasks"]) == {"literature", "summarizer", "methodology", "code_builder"}


def check_supervisor_fallback_uses_minimal_plan() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {}

    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(_state(query="请总结并分析这篇论文的方法"))

    assert update["next_agent"] == "FINISH"
    assert update["task_plan"] == []


def check_supervisor_uses_llm_plan_without_keyword_trimming() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {
            "task_plan": ["literature", "summarizer", "methodology", "code_builder"],
            "module_tasks": {},
            "reason": "LLM selected the full plan",
            "skills": [],
            "direct_answer": "",
        }

    state = _state(query="复现代码并不完整，并且没有实际运行验证")
    state["memory"]["recent_code_project_path"] = "D:/scholar code/demo"
    state["memory"]["recent_code_delivery_status"] = "incomplete"

    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(state)

    assert update["task_plan"] == ["literature", "summarizer", "methodology", "code_builder"]


def check_supervisor_uses_module_decisions_over_inconsistent_task_plan() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {
            "task_plan": ["literature", "summarizer", "methodology", "code_builder"],
            "module_decisions": {
                "literature": {"selected": False, "reason": "论文已指定，不需要检索候选论文"},
                "summarizer": {"selected": True, "reason": "需要总结论文"},
                "methodology": {"selected": True, "reason": "需要介绍方法"},
                "code_builder": {"selected": True, "reason": "需要项目复现代码"},
            },
            "module_tasks": {},
            "reason": "无需文献检索，按序总结、方法分析、代码构建。",
            "skills": [],
            "direct_answer": "",
        }

    query = (
        "总结一下An_Improved_EPA-Based_Receiver_Design_for_Uplink_"
        "LDPC_Coded_SCMA_System论文，并介绍其中的方法，给出完整的项目复现代码"
    )
    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(_state(query=query))

    assert update["task_plan"] == ["summarizer", "methodology", "code_builder"]
    assert "literature" not in update["module_tasks"]


def check_supervisor_routes_database_management() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {
            "task_plan": [],
            "module_decisions": {
                "literature": {"selected": False, "reason": "不需要检索论文"},
                "summarizer": {"selected": False, "reason": "不需要总结论文"},
                "methodology": {"selected": False, "reason": "不需要方法分析"},
                "code_builder": {"selected": False, "reason": "不需要代码"},
                "database_manager": {"selected": True, "reason": "需要删除本地数据库记录"},
            },
            "module_tasks": {
                "database_manager": "删除本地数据库中 ID 为 local_75ea7b80 的论文记录。"
            },
            "reason": "用户要求管理本地数据库。",
            "skills": [],
            "direct_answer": "",
        }

    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(_state(query="删除本地数据库中ID: local_75ea7b80的数据"))

    assert update["next_agent"] == "module_executor"
    assert update["task_plan"] == ["database_manager"]
    assert "database_manager" in update["module_tasks"]


def check_supervisor_routes_writing_editor() -> None:
    def fake_plan(**kwargs: Any) -> dict:
        return {
            "task_plan": [],
            "module_decisions": {
                "literature": {"selected": False, "reason": "不需要检索论文"},
                "summarizer": {"selected": False, "reason": "不需要总结论文"},
                "methodology": {"selected": False, "reason": "不需要方法分析"},
                "code_builder": {"selected": False, "reason": "不需要代码"},
                "database_manager": {"selected": False, "reason": "不需要管理数据库"},
                "writing_editor": {"selected": True, "reason": "需要润色英文论文 docx 文稿"},
            },
            "module_tasks": {
                "writing_editor": "读取 draft/paper.docx 并按 IEEE 英文学术风格润色。"
            },
            "reason": "用户要求论文写作润色。",
            "skills": [],
            "direct_answer": "",
        }

    with _patched(supervisor_mod, "_plan_with_llm", fake_plan):
        update = supervisor_node(_state(query="请润色 draft/paper.docx，英文 IEEE 风格"))

    assert update["next_agent"] == "module_executor"
    assert update["task_plan"] == ["writing_editor"]
    assert "writing_editor" in update["module_tasks"]


def check_compiled_graph_reaches_finish_with_fake_nodes() -> None:
    def fake_supervisor(state: dict) -> dict:
        return {
            "next_agent": "module_executor",
            "task_plan": ["literature", "summarizer"],
            "module_tasks": {
                "literature": "fake literature task",
                "summarizer": "fake summarizer task",
            },
            "current_task": "fake modular plan",
            "messages": [HumanMessage(content="Supervisor 模块计划: fake", name="supervisor")],
        }

    def fake_module_executor(state: dict) -> dict:
        return {
            "next_agent": "synthesis",
            "expert_outputs": [
                {"expert_name": "literature", "content": state["module_tasks"]["literature"]},
                {"expert_name": "summarizer", "content": state["module_tasks"]["summarizer"]},
            ],
            "messages": [
                HumanMessage(content="literature done", name="literature"),
                HumanMessage(content="summarizer done", name="summarizer"),
            ],
        }

    def fake_synthesis(state: dict) -> dict:
        return {
            "next_agent": "FINISH",
            "messages": [HumanMessage(content="fake graph final", name="supervisor")],
        }

    with _patched(graph_mod, "supervisor_node", fake_supervisor):
        with _patched(graph_mod, "module_executor_node", fake_module_executor):
            with _patched(graph_mod, "synthesis_node", fake_synthesis):
                compiled = graph_mod.build_multi_agent_graph()

    result = compiled.invoke(
        _state(query="fake graph request"),
        config={"configurable": {"thread_id": "manual_workflow_check"}, "recursion_limit": 10},
    )

    assert result["next_agent"] == "FINISH"
    assert [item["expert_name"] for item in result["expert_outputs"]] == ["literature", "summarizer"]
    assert result["messages"][-1].content == "fake graph final"


def check_code_builder_uses_selected_project_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        selected_project = workspace / "paper_local_55ac1755_a1109e24"
        selected_project.mkdir()
        (selected_project / "src").mkdir()

        selected_state = _state(query="请阅读已有代码")
        project_path, project_slug = code_builder_mod._select_code_project(
            selected_state,
            str(selected_project),
            "read existing project",
            "read existing project",
            "",
        )
        assert Path(project_path) == selected_project.resolve()
        assert project_slug == selected_project.name
        assert Path(project_path) != selected_project / "paper_reproduction_2fd9170b"


def check_code_builder_default_workspace_creates_project_folder() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_default = code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR
        try:
            code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR = tmpdir
            state = _state(query="请生成代码")
            path, slug = code_builder_mod._select_code_project(
                state,
                tmpdir,
                "paper reproduction",
                "paper reproduction",
                "methodology",
            )
            assert Path(path).parent == Path(tmpdir).resolve()
            assert Path(path).name == slug
            assert Path(path) != Path(tmpdir).resolve()
        finally:
            code_builder_mod.conf.CODE_BUILDER_WORKSPACE_DIR = original_default


def main() -> None:
    checks = [
        ("task_plan_inference", check_task_plan_inference),
        ("default_workspace_layout", check_default_workspace_layout),
        ("plan_normalization", check_plan_normalization),
        ("module_executor_runs_plan_in_order", check_module_executor_runs_plan_in_order),
        ("module_executor_skips_downstream_after_no_papers", check_module_executor_skips_downstream_after_no_papers),
        ("expert_output_reducer", check_expert_output_reducer),
        ("paper_selection_helpers", check_paper_selection_helpers),
        ("graph_route_sanitization", check_graph_route_sanitization),
        ("latest_user_query_and_no_papers_message", check_latest_user_query_and_no_papers_message),
        ("search_tool_schema_guides_llm_query_arguments", check_search_tool_schema_guides_llm_query_arguments),
        ("literature_tool_call_dispatch_uses_llm_arguments", check_literature_tool_call_dispatch_uses_llm_arguments),
        ("database_manager_deletes_with_tool_result", check_database_manager_deletes_with_tool_result),
        ("writing_document_tools_docx_and_tex", check_writing_document_tools_docx_and_tex),
        ("synthesis_does_not_claim_missing_writing_documents", check_synthesis_does_not_claim_missing_writing_documents),
        ("code_builder_quality_review_revision", check_code_builder_quality_review_revision),
        ("code_builder_prompt_matches_autonomous_project_workflow", check_code_builder_prompt_matches_autonomous_project_workflow),
        ("code_workspace_tools_write_and_run", check_code_workspace_tools_write_and_run),
        ("service_initial_state_accepts_code_python_executable", check_service_initial_state_accepts_code_python_executable),
        ("code_builder_includes_all_external_mcp_tools", check_code_builder_includes_all_external_mcp_tools),
        ("code_builder_autonomous_loop_uses_tools", check_code_builder_autonomous_loop_uses_tools),
        ("code_builder_defers_final_until_files_and_validation", check_code_builder_defers_final_until_files_and_validation),
        ("code_builder_requires_all_validation_plan_targets", check_code_builder_requires_all_validation_plan_targets),
        ("code_builder_continues_recent_incomplete_project", check_code_builder_continues_recent_incomplete_project),
        ("code_builder_cancels_unvalidated_project_on_shutdown", check_code_builder_cancels_unvalidated_project_on_shutdown),
        ("code_builder_has_no_tool_round_budget", check_code_builder_has_no_tool_round_budget),
        ("code_builder_ignores_unexpected_tool_arguments", check_code_builder_ignores_unexpected_tool_arguments),
        ("code_builder_uses_selected_project_root", check_code_builder_uses_selected_project_root),
        ("code_builder_default_workspace_creates_project_folder", check_code_builder_default_workspace_creates_project_folder),
        ("memory_store_roundtrip_and_prompt", check_memory_store_roundtrip_and_prompt),
        ("logging_writes_file", check_logging_writes_file),
        ("service_loads_and_updates_memory", check_service_loads_and_updates_memory),
        ("final_answer_filter", check_final_answer_filter),
        ("supervisor_direct_response_without_modules", check_supervisor_direct_response_without_modules),
        ("supervisor_plans_modules_once_without_chain", check_supervisor_plans_modules_once_without_chain),
        ("supervisor_fallback_uses_minimal_plan", check_supervisor_fallback_uses_minimal_plan),
        ("supervisor_uses_llm_plan_without_keyword_trimming", check_supervisor_uses_llm_plan_without_keyword_trimming),
        (
            "supervisor_uses_module_decisions_over_inconsistent_task_plan",
            check_supervisor_uses_module_decisions_over_inconsistent_task_plan,
        ),
        ("supervisor_routes_database_management", check_supervisor_routes_database_management),
        ("supervisor_routes_writing_editor", check_supervisor_routes_writing_editor),
        ("compiled_graph_reaches_finish_with_fake_nodes", check_compiled_graph_reaches_finish_with_fake_nodes),
    ]
    for name, fn in checks:
        _run_check(name, fn)
    print(f"OK {len(checks)} checks passed")


if __name__ == "__main__":
    main()
