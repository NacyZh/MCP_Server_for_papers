import importlib.util
import sys
from pathlib import Path

import pytest


_MISSING_DEPS = [
    package
    for package in ("langchain_core", "langgraph", "langchain_openai")
    if importlib.util.find_spec(package) is None
]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing optional agent workflow dependencies: {', '.join(_MISSING_DEPS)}",
)

if not _MISSING_DEPS:
    from langchain_core.messages import HumanMessage

    from scholar_agent.agents import executor as executor_mod
    from scholar_agent.agents import synthesis as synthesis_mod
    from scholar_agent.agents import supervisor as supervisor_mod
    from scholar_agent.agents.experts import code_builder as code_builder_mod
    from scholar_agent.agents.experts import database_manager as database_manager_mod
    from scholar_agent.agents.experts import literature as literature_mod
    from scholar_agent.agents.graph import _route_supervisor
    from scholar_agent.agents.planning import infer_task_plan
    from scholar_agent.agents.service import MultiAgentService, _is_final_supervisor_message
    from scholar_agent.agents.state import append_expert_outputs
    from scholar_agent.agents.supervisor import supervisor_node
    from scholar_agent.agents.synthesis import synthesis_node
    from scholar_agent.agents.utils import (
        collect_paper_ids_from_outputs,
        collect_retrieval_query_from_outputs,
        filter_chunks_by_paper_ids,
    )
    from scholar_agent.agents.utils import no_papers_response
    from scholar_agent.core.runtime import clear_shutdown_request, request_shutdown
    from scholar_agent.config import conf
    from scholar_agent.tools import code_tools as code_tools_mod
    from scholar_agent.tools import writing_tools as writing_tools_mod
    from scholar_agent.tools.paper_tools import ArxivSearchTool, LocalSearchTool
    from scholar_agent.tools.registry import get_tool_registry
    from scholar_agent.prompts import CODE_BUILDER_SYSTEM_PROMPT


def test_infer_task_plan_uses_minimal_modules_by_default():
    assert infer_task_plan("请根据论文方法生成 MATLAB 复现代码") == []
    assert infer_task_plan("请完成完整研究任务，从文献到代码") == []
    assert infer_task_plan("总结这篇论文") == []


def test_default_workspace_layout_is_unified():
    work_root = Path(conf.SCHOLAR_AGENT_WORK_ROOT)
    code_root = Path(conf.CODE_BUILDER_WORKSPACE_DIR)
    document_root = Path(conf.WRITING_WORKSPACE_DIR)

    assert work_root.as_posix().endswith("scholar agent")
    assert code_root.parent == work_root
    assert document_root.parent == work_root
    assert code_root.name == "scholar code"
    assert document_root.name == "scholar document"


def _state_with_runtime(**updates):
    state = {
        "messages": [HumanMessage(content="test")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "test",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": conf.CODE_BUILDER_WORKSPACE_DIR,
        "code_workspace_is_project": False,
        "code_python_executable": conf.CODE_BUILDER_PYTHON_EXECUTABLE,
    }
    state.update(updates)
    return state


def test_expert_outputs_accumulate_with_reducer():
    existing = [{"expert_name": "literature", "content": "a"}]
    updates = [{"expert_name": "summarizer", "content": "b"}]

    assert append_expert_outputs(existing, updates) == existing + updates


def test_paper_selection_helpers_keep_prior_selection():
    state = {
        "expert_outputs": [
            {"expert_name": "literature", "metadata": {"paper_ids": ["p1", "p2"]}},
            {"expert_name": "summarizer", "metadata": {"paper_ids": ["p2", "p3"]}},
        ]
    }
    assert collect_paper_ids_from_outputs(state, preferred_experts=["literature"]) == ["p1", "p2"]
    chunks = [{"paper_id": "p0"}, {"paper_id": "p2"}]
    assert filter_chunks_by_paper_ids(chunks, ["p2"]) == [chunks[1]]


def test_downstream_modules_reuse_literature_retrieval_query():
    state = {
        "expert_outputs": [
            {
                "expert_name": "literature",
                "metadata": {
                    "retrieval_query": "all:\"scma\" AND all:\"detection\"",
                    "local_query": "SCMA detection message passing deep learning receiver",
                },
            }
        ]
    }

    assert collect_retrieval_query_from_outputs(state, preferred_experts=["literature"]) == (
        "SCMA detection message passing deep learning receiver"
    )


def test_retrieval_query_from_context_does_not_use_task_text():
    from scholar_agent.agents.utils import retrieval_query_from_context

    state = {
        "memory": {
            "recent_topics": ["导入本地论文"],
            "recent_paper_ids": ["local_abc123"],
        }
    }
    user_task = "总结论文并生成一份中文 Word 和一份英文 LaTeX 文档"

    query = retrieval_query_from_context(
        state,
        prior_query="",
        fallback_query="paper summary",
    )

    assert "paper summary" in query
    assert "local_abc123" in query
    assert user_task not in query


def test_supervisor_direct_response_without_modules(monkeypatch):
    monkeypatch.setattr(
        supervisor_mod,
        "_plan_with_llm",
        lambda **kwargs: {
            "task_plan": [],
            "module_tasks": {},
            "reason": "no module needed",
            "skills": [],
            "direct_answer": "直接回复，不调用学术模块。",
        },
    )
    state = {
        "messages": [HumanMessage(content="任意短输入")],
        "session_id": "pytest-session",
        "memory": {
            "session_id": "pytest-session",
            "summary": "",
            "user_preferences": [],
            "recent_topics": [],
            "recent_paper_ids": [],
            "active_skills": [],
            "turn_count": 0,
        },
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = supervisor_node(state)

    assert update["next_agent"] == "FINISH"
    assert update["task_plan"] == []
    assert update["messages"][0].content == "直接回复，不调用学术模块。"


def test_supervisor_plans_module_executor(monkeypatch):
    monkeypatch.setattr(
        supervisor_mod,
        "_plan_with_llm",
        lambda **kwargs: {
            "task_plan": ["summarizer", "methodology"],
            "module_tasks": {},
            "reason": "LLM selected summary and methodology",
            "skills": [],
            "direct_answer": "",
        },
    )
    state = {
        "messages": [HumanMessage(content="请总结并分析这篇论文的方法")],
        "session_id": "pytest-session",
        "memory": {
            "session_id": "pytest-session",
            "summary": "",
            "user_preferences": [],
            "recent_topics": [],
            "recent_paper_ids": [],
            "active_skills": [],
            "turn_count": 0,
        },
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = supervisor_node(state)

    assert update["next_agent"] == "module_executor"
    assert update["task_plan"] == ["summarizer", "methodology"]
    assert "Supervisor 模块计划:" in update["messages"][0].content


def test_supervisor_uses_llm_plan_without_keyword_trimming(monkeypatch):
    monkeypatch.setattr(
        supervisor_mod,
        "_plan_with_llm",
        lambda **kwargs: {
            "task_plan": ["literature", "summarizer", "methodology", "code_builder"],
            "module_tasks": {},
            "reason": "LLM selected the full plan",
            "skills": [],
            "direct_answer": "",
        },
    )
    state = {
        "messages": [HumanMessage(content="复现代码并不完整，并且没有实际运行验证")],
        "session_id": "pytest-session",
        "memory": {
            "session_id": "pytest-session",
            "summary": "",
            "user_preferences": [],
            "recent_topics": [],
            "recent_paper_ids": [],
            "active_skills": [],
            "recent_code_project_path": "D:/scholar code/demo",
            "recent_code_delivery_status": "incomplete",
            "turn_count": 1,
        },
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = supervisor_node(state)

    assert update["task_plan"] == ["literature", "summarizer", "methodology", "code_builder"]


def test_supervisor_uses_module_decisions_over_inconsistent_task_plan(monkeypatch):
    monkeypatch.setattr(
        supervisor_mod,
        "_plan_with_llm",
        lambda **kwargs: {
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
        },
    )
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "总结一下An_Improved_EPA-Based_Receiver_Design_for_Uplink_"
                    "LDPC_Coded_SCMA_System论文，并介绍其中的方法，给出完整的项目复现代码"
                )
            )
        ],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = supervisor_node(state)

    assert update["task_plan"] == ["summarizer", "methodology", "code_builder"]
    assert "literature" not in update["module_tasks"]


def test_supervisor_routes_database_management(monkeypatch):
    monkeypatch.setattr(
        supervisor_mod,
        "_plan_with_llm",
        lambda **kwargs: {
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
        },
    )
    state = {
        "messages": [HumanMessage(content="删除本地数据库中ID: local_75ea7b80的数据")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = supervisor_node(state)

    assert update["next_agent"] == "module_executor"
    assert update["task_plan"] == ["database_manager"]
    assert "database_manager" in update["module_tasks"]


def test_supervisor_routes_writing_editor(monkeypatch):
    monkeypatch.setattr(
        supervisor_mod,
        "_plan_with_llm",
        lambda **kwargs: {
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
        },
    )
    state = {
        "messages": [HumanMessage(content="请润色 draft/paper.docx，英文 IEEE 风格")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = supervisor_node(state)

    assert update["next_agent"] == "module_executor"
    assert update["task_plan"] == ["writing_editor"]
    assert "writing_editor" in update["module_tasks"]


def test_graph_route_sanitizes_finish_and_invalid_values():
    assert _route_supervisor({"next_agent": "finish"}) == "FINISH"
    assert _route_supervisor({"next_agent": "Literature"}) == "FINISH"
    assert _route_supervisor({"next_agent": "Literature", "task_plan": ["literature"]}) == "module_executor"
    assert _route_supervisor({"next_agent": "not-an-agent"}) == "FINISH"


def test_service_final_answer_filter_ignores_routing_messages():
    routing = HumanMessage(content="Supervisor 调度: go literature", name="supervisor")
    module_plan = HumanMessage(content="Supervisor 模块计划: run modules", name="supervisor")
    final = HumanMessage(content="最终整合答案", name="supervisor")

    assert not _is_final_supervisor_message(routing)
    assert not _is_final_supervisor_message(module_plan)
    assert _is_final_supervisor_message(final)


def test_search_tool_schema_guides_llm_query_arguments():
    local_text = f"{LocalSearchTool.description} {LocalSearchTool.params['properties']['query']['description']}"
    arxiv_text = f"{ArxivSearchTool.description} {ArxivSearchTool.params['properties']['query']['description']}"

    assert "planning text" in local_text
    assert "Concise academic search query" in local_text
    assert "Concise English search keywords" in arxiv_text
    assert "Translate the research topic to English" in arxiv_text


def test_literature_tool_call_dispatch_uses_llm_arguments():
    class FakeToolResult:
        def __init__(self, status="success", result="ok", data=None):
            self.status = status
            self.result = result
            self.data = data

    class FakeTool:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return FakeToolResult(data=[])

    class FakeRegistry:
        def __init__(self):
            self.tools = {
                "list_local_database": FakeTool(),
                "search_local_database": FakeTool(),
                "search_arxiv_papers": FakeTool(),
                "search_local_papers_chunks": FakeTool(),
            }

        def get(self, name):
            return self.tools[name]

        def to_langchain_tool(self, name):
            return self.tools[name]

    class FakeBoundLLM:
        def invoke(self, messages):
            return type(
                "FakeResponse",
                (),
                {
                    "tool_calls": [
                        {
                            "name": "search_arxiv_papers",
                            "args": {"query": "SCMA detection message passing", "max_results": 3},
                        }
                    ]
                },
            )()

    class FakeLLM:
        def bind_tools(self, tools):
            return FakeBoundLLM()

    registry = FakeRegistry()
    task = "围绕用户问题检索本地数据库和 arXiv，筛选最相关论文: 请围绕 SCMA 完成完整研究任务"
    literature_mod._run_literature_tool_calls(FakeLLM(), registry, "请检索 SCMA 检测相关论文", task, {})

    assert registry.tools["search_arxiv_papers"].calls == [
        {"query": "SCMA detection message passing", "max_results": 3}
    ]
    assert task not in str(registry.tools["search_arxiv_papers"].calls)


def test_database_manager_deletes_with_tool_result(monkeypatch):
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

        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return FakeToolResult()

    class FakeRegistry:
        def __init__(self):
            self.tool = FakeTool()

        def get(self, name):
            return self.tool if name == "delete_paper_from_database" else None

        def to_langchain_tool(self, name):
            return self.tool

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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
        def __init__(self):
            self.bound = FakeBoundLLM()

        def bind_tools(self, tools):
            return self.bound

    registry = FakeRegistry()
    monkeypatch.setattr(database_manager_mod, "get_tool_registry", lambda: registry)
    monkeypatch.setattr(database_manager_mod, "ChatOpenAI", lambda **kwargs: FakeLLM())

    update = database_manager_mod.database_manager_node(
        _state_with_runtime(
            messages=[HumanMessage(content="删除本地数据库中ID: local_75ea7b80的数据")],
            current_task="删除本地数据库中ID: local_75ea7b80的数据",
            code_workspace_path="D:/scholar code",
        )
    )

    assert registry.tool.calls == [{"local_id": "local_75ea7b80"}]
    output = update["expert_outputs"][0]
    assert output["expert_name"] == "database_manager"
    assert output["metadata"]["tool_calls"][0]["tool"] == "delete_paper_from_database"
    assert output["metadata"]["tool_calls"][0]["status"] == "success"
    assert "Deleted paper local_75ea7b80" in output["content"]


def test_writing_document_tools_docx_and_tex(monkeypatch, tmp_path):
    monkeypatch.setattr(writing_tools_mod.conf, "WRITING_WORKSPACE_DIR", str(tmp_path))

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

    compile_result = writing_tools_mod.WritingCompileLatexDocumentTool().execute(path="draft/main.tex", engine="auto")
    assert compile_result.status in {"success", "fail"}
    if compile_result.status == "success":
        assert "PDF:" in compile_result.result
    else:
        assert "LaTeX" in compile_result.result


def test_synthesis_does_not_claim_missing_writing_documents(monkeypatch):
    class FakeLLM:
        def __init__(self, **kwargs):
            raise AssertionError("synthesis LLM should not run for failed writing delivery")

    monkeypatch.setattr(synthesis_mod, "ChatOpenAI", FakeLLM)
    state = {
        "messages": [HumanMessage(content="总结SCMA论文并生成中文Word和英文LaTeX")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "synthesis",
        "expert_outputs": [
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
        ],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = synthesis_node(state)
    answer = update["messages"][0].content

    assert "文档生成未完成" in answer
    assert "没有任何 `writing_write_docx_document`" in answer
    assert "已生成" not in answer
    assert "writing_output/SCMA_Literature_Summary_中文.docx" not in answer


def test_module_executor_skips_downstream_after_no_papers(monkeypatch):
    calls = []

    def fake_literature(state):
        calls.append("literature")
        return no_papers_response("literature")

    def fake_code_builder(state):
        calls.append("code_builder")
        return {"expert_outputs": [{"expert_name": "code_builder", "content": "bad"}]}

    monkeypatch.setattr(
        executor_mod,
        "_EXPERT_NODES",
        {"literature": fake_literature, "code_builder": fake_code_builder},
    )
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": ["literature", "code_builder"],
        "module_tasks": {},
        "current_task": "",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = executor_mod.module_executor_node(state)

    assert calls == ["literature"]
    assert [item["expert_name"] for item in update["expert_outputs"]] == ["literature"]


def test_code_builder_quality_review_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "CODE_BUILDER_WORKSPACE_DIR", str(tmp_path))

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse("初版代码")
            if self.calls == 2:
                return FakeResponse("## 质量结论\nNEEDS_REVISION\n\n## 必须修改\n- 补入口脚本")
            return FakeResponse("修订后完整代码")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "生成代码",
        "active_skills": [],
        "skill_overrides": {},
    }

    update = code_builder_mod.code_builder_node(state)

    output = update["expert_outputs"][0]
    assert output["content"] == "修订后完整代码"
    assert output["metadata"]["quality_reviewed"] is True
    assert output["metadata"]["revised_after_review"] is True


def test_code_builder_prompt_matches_autonomous_project_workflow():
    prompt = CODE_BUILDER_SYSTEM_PROMPT

    assert "项目级自主编码代理" in prompt
    assert "code_workspace_set_validation_plan" in prompt
    assert "code_workspace_record_validation" in prompt
    assert "最后一次文件修改后" in prompt
    assert "不要直接输出大段代码当作完成" in prompt


def test_code_workspace_tools_are_registered_and_sandboxed(tmp_path, monkeypatch):
    monkeypatch.setattr(code_tools_mod, "CODE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(code_tools_mod.conf, "CODE_BUILDER_PYTHON_EXECUTABLE", sys.executable)
    registry = get_tool_registry()

    assert registry.get("code_workspace_write_file") is not None
    assert registry.get("code_workspace_run_python") is not None
    assert registry.get("code_workspace_run_shell") is not None

    write_result = registry.get("code_workspace_write_file").execute(
        path="demo/main.py",
        content="print('ok')\n",
    )
    assert write_result.status == "success"

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

    blocked_shell = registry.get("code_workspace_run_shell").execute(command="Remove-Item -Recurse -Force .")
    assert blocked_shell.status == "fail"
    assert "blocked" in blocked_shell.result.lower()

    with code_tools_mod.use_code_python_executable(sys.executable) as selected_python:
        assert Path(selected_python).resolve() == Path(sys.executable).resolve()
        contextual_run = registry.get("code_workspace_run_shell").execute(
            command="python -c \"import sys; print(sys.executable)\"",
            cwd="demo",
            timeout_s=10,
        )
    assert contextual_run.status == "success"
    assert Path(contextual_run.result.strip().splitlines()[-1]).resolve() == Path(sys.executable).resolve()

    patch_result = registry.get("code_workspace_make_patch").execute(
        path="demo/main.py",
        new_content="print('patched')\n",
    )
    assert patch_result.status == "success"
    assert "--- a/demo/main.py" in patch_result.result
    apply_result = registry.get("code_workspace_apply_patch").execute(patch=patch_result.result)
    assert apply_result.status == "success"
    assert (tmp_path / "demo" / "main.py").read_text(encoding="utf-8") == "print('patched')\n"

    escape_result = registry.get("code_workspace_write_file").execute(path="../bad.py", content="")
    assert escape_result.status == "fail"

    bad_patch = "--- a/../bad.py\n+++ b/../bad.py\n@@ -0,0 +1 @@\n+x\n"
    bad_patch_result = registry.get("code_workspace_apply_patch").execute(patch=bad_patch)
    assert bad_patch_result.status == "fail"


def test_code_builder_includes_all_external_mcp_tools():
    class FakeTool:
        def __init__(self, is_external=False):
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

        def list_all(self):
            return sorted(self.tools)

        def get(self, name):
            return self.tools.get(name)

    names = code_builder_mod._available_code_builder_tool_names(FakeRegistry())

    assert "code_workspace_write_file" in names
    assert "code_workspace_run_shell" in names
    assert "github__create_issue" in names
    assert "jupyter__run_cell" in names
    assert "search_local_papers_chunks" not in names


def test_service_initial_state_accepts_code_python_executable(tmp_path):
    service = MultiAgentService()
    state = service._build_initial_state(
        "生成代码",
        session_id="pytest-runtime",
        code_workspace_path=str(tmp_path),
        code_workspace_is_project=True,
        code_python_executable=sys.executable,
    )

    assert state["code_workspace_path"] == str(tmp_path)
    assert state["code_workspace_is_project"] is True
    assert Path(state["code_python_executable"]).resolve() == Path(sys.executable).resolve()


def test_code_builder_autonomous_loop_uses_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(code_tools_mod, "CODE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.plain_calls += 1
            if self.plain_calls == 1:
                return FakeResponse("## 质量结论\nPASS")
            return FakeResponse("unused")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "生成代码",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    update = code_builder_mod.code_builder_node(state)
    output = update["expert_outputs"][0]

    assert "最终说明" in output["content"]
    assert output["metadata"]["autonomous_coding"] is True
    assert "code_workspace_apply_patch" in output["metadata"]["tool_trace"]
    assert "code_workspace_run_python" in output["metadata"]["tool_trace"]
    project_path = Path(output["metadata"]["code_project_path"])
    assert project_path == tmp_path.resolve()
    assert (project_path / "demo" / "main.py").exists()


def test_code_builder_defers_final_until_files_and_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return FakeResponse("## 质量结论\nPASS")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "生成代码",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    update = code_builder_mod.code_builder_node(state)
    output = update["expert_outputs"][0]

    assert output["content"] == "最终说明: 已创建并验证。"
    assert fake_llm.bound.calls == 5
    assert "code_workspace_write_file" in output["metadata"]["tool_trace"]
    assert "code_workspace_set_validation_plan" in output["metadata"]["tool_trace"]
    assert "code_workspace_run_python" in output["metadata"]["tool_trace"]
    assert "code_workspace_record_validation" not in output["metadata"]["tool_trace"]


def test_code_builder_requires_all_validation_plan_targets(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return FakeResponse("## 质量结论\nPASS")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="运行 test_components.m 和 run_simulation.m 直到无错")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "运行 test_components.m 和 run_simulation.m 直到无错",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    update = code_builder_mod.code_builder_node(state)
    output = update["expert_outputs"][0]

    assert output["content"] == "最终说明: 两个 MATLAB 脚本均已通过。"
    assert fake_llm.bound.calls == 6
    assert "code_workspace_set_validation_plan" in output["metadata"]["tool_trace"]
    assert output["metadata"]["tool_trace"].count("code_workspace_record_validation") == 2


def test_code_builder_continues_recent_incomplete_project(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)
    monkeypatch.setattr(code_builder_mod.conf, "CODE_BUILDER_WORKSPACE_DIR", str(tmp_path))
    recent_project = tmp_path / "paper_local_123"
    recent_project.mkdir()

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return FakeResponse("## 质量结论\nPASS")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="继续运行并修复验证脚本")],
        "session_id": "pytest-session",
        "memory": {
            "recent_code_project_path": str(recent_project),
            "recent_code_delivery_status": "incomplete",
        },
        "next_agent": "supervisor",
        "expert_outputs": [],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "继续运行并修复验证脚本",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    update = code_builder_mod.code_builder_node(state)
    output = update["expert_outputs"][0]

    assert Path(output["metadata"]["code_project_path"]) == recent_project.resolve()
    assert output["metadata"]["code_project_slug"] == recent_project.name
    assert (recent_project / "main.py").exists()


def test_code_builder_cancels_unvalidated_project_on_shutdown(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.plain_calls += 1
            return FakeResponse("不应调用质量审查")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "生成代码",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    try:
        update = code_builder_mod.code_builder_node(state)
        output = update["expert_outputs"][0]
    finally:
        clear_shutdown_request()

    assert output["metadata"]["delivery_status"] == "cancelled"
    assert "代码构建已中止" in output["content"]
    assert fake_llm.plain_calls == 0


def test_code_builder_has_no_tool_round_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return FakeResponse("unused")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "生成代码",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    update = code_builder_mod.code_builder_node(state)
    output = update["expert_outputs"][0]

    assert fake_llm.bound.calls == 97
    assert output["metadata"]["delivery_status"] == "complete"
    assert output["content"] == "最终说明: 已完成长轮次调试。"


def test_code_builder_ignores_unexpected_tool_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(code_builder_mod.conf, "AGENT_CODE_BUILDER_AUTONOMOUS", True)

    class FakeResponse:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeBoundLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
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

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return FakeResponse("## 质量结论\nPASS")

    fake_llm = FakeLLM()
    monkeypatch.setattr(code_builder_mod, "ChatOpenAI", lambda **kwargs: fake_llm)
    state = {
        "messages": [HumanMessage(content="请生成代码")],
        "session_id": "pytest-session",
        "memory": {},
        "next_agent": "supervisor",
        "expert_outputs": [{"expert_name": "methodology", "content": "方法规格", "metadata": {"paper_ids": ["p1"]}}],
        "task_plan": [],
        "module_tasks": {},
        "current_task": "生成代码",
        "active_skills": [],
        "skill_overrides": {},
        "code_workspace_path": str(tmp_path),
    }

    update = code_builder_mod.code_builder_node(state)
    output = update["expert_outputs"][0]

    assert output["content"] == "最终说明: 已创建并验证。"
    assert "Ignored unexpected argument(s): content" in output["metadata"]["tool_trace"]
