import sys
import threading

from rag.tools.base import ToolResult, execute_tool_safely
from rag.tools.jobs import job_manager
from rag.tools.paper_tools import (
    BuildPaperSummaryTool,
    DbAddTool,
    DbImportDirectoryTool,
    DbListTool,
    DbSearchTool,
    EvidenceChunkRetrievalTool,
    PaperOutlineTool,
    PaperProfileTool,
    PaperSummaryTool,
    RagHealthCheckTool,
    RetrievalQualityEvalTool,
    ToolJobStatusTool,
    build_default_tools,
    summary_job_manager,
)


class FakePaperDB:
    def get_all_papers(self):
        return [{"id": "local_1", "title": "Demo Paper"}]

    def search_papers(self, keyword):
        assert keyword == "attention"
        return [{
            "id": "local_2",
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "publish_year": 2017,
            "tags": "transformer",
            "sections": ["Introduction (2)"],
        }]

    def get_paper_sections(self, paper_id):
        assert paper_id == "local_1"
        return ["Introduction (2)", "Method (3)"]


def test_list_local_database_does_not_import_vector_store(monkeypatch):
    sys.modules.pop("rag.storage.vector_store", None)
    sys.modules.pop("rag.plugins.pdf_parser", None)

    result = DbListTool(paper_db=FakePaperDB()).execute()

    assert result.status == "success"
    assert "local_1" in result.result
    assert "rag.storage.vector_store" not in sys.modules
    assert "rag.plugins.pdf_parser" not in sys.modules


def test_search_local_database_does_not_import_vector_store():
    sys.modules.pop("rag.storage.vector_store", None)
    sys.modules.pop("rag.plugins.pdf_parser", None)

    result = DbSearchTool(paper_db=FakePaperDB()).execute(query="attention")

    assert result.status == "success"
    assert "local_2" in result.result
    assert "Attention Is All You Need" in result.result
    assert result.data["papers"][0]["id"] == "local_2"
    assert "rag.storage.vector_store" not in sys.modules
    assert "rag.plugins.pdf_parser" not in sys.modules


def test_default_tools_include_local_database_list_and_search(monkeypatch):
    tools = build_default_tools()

    assert "list_local_database" in tools
    assert "search_local_database" in tools
    assert "retrieve_evidence_chunks" in tools
    assert "get_paper_outline" in tools
    assert "get_paper_profile" in tools
    assert "get_paper_summary" in tools
    assert "build_paper_summary" in tools
    assert "add_paper_to_database" in tools
    assert "import_papers_from_directory" in tools
    assert "get_tool_job_status" in tools
    assert "rag_health_check" in tools
    assert "delete_paper_from_database" in tools
    assert "dedup_local_database" in tools
    assert "backfill_paper_metadata" in tools
    assert "evaluate_retrieval_quality" in tools
    assert "writing_write_docx_document" not in tools
    assert "code_workspace_run_python" not in tools


def test_default_tools_are_fixed_rag_tool_set():
    tools = build_default_tools()

    assert "list_local_database" in tools
    assert "search_local_database" in tools
    assert "retrieve_evidence_chunks" in tools
    assert "get_paper_outline" in tools
    assert "get_paper_profile" in tools
    assert "get_paper_summary" in tools
    assert "build_paper_summary" in tools
    assert "add_paper_to_database" in tools
    assert "import_papers_from_directory" in tools
    assert "get_tool_job_status" in tools
    assert "rag_health_check" in tools
    assert "evaluate_retrieval_quality" in tools
    assert "writing_write_docx_document" not in tools
    assert "code_workspace_run_python" not in tools


class FakePaperManager:
    def get_chunks_for_paper_ids(self, ids, max_chunks_per_paper=5, section=""):
        assert ids == ["local_1"]
        assert section == "Introduction"
        return [{
            "paper_id": "local_1",
            "title": "Demo",
            "section_name": "introduction",
            "section_title": "Introduction",
            "content": "Intro text",
        }]

    def search_knowledge(self, query, n_results=5):
        assert query == "attention"
        return [{
            "paper_id": "local_2",
            "title": "Attention",
            "chunk_id": "local_2:0",
            "score": 0.9,
            "section_name": "method",
            "section_title": "Method",
            "content": "Method text",
        }]


class FakeSummaryDB:
    def __init__(self):
        self.built = False

    def get_paper_outline(self, paper_id, language="en", detail_level="medium"):
        assert paper_id == "local_1"
        assert language == "en"
        assert detail_level == "medium"
        return {
            "paper_id": "local_1",
            "title": "Demo",
            "sections": [{
                "section_id": "0001_introduction",
                "section": "Introduction",
                "normalized_section_name": "introduction",
                "section_order": 0,
                "chunk_count": 2,
                "char_count": 1200,
                "content_hash": "abc",
                "has_summary": self.built,
            }],
        }

    def get_profile(self, paper_id, language):
        if not self.built:
            return None
        return {
            "paper_id": paper_id,
            "language": language,
            "title": "Demo",
            "status": "ready",
            "problem": "Problem",
            "background": "Background",
            "method": "Method",
            "contributions": ["Contribution"],
            "experiments": "Experiments",
            "limitations": "Limitations",
            "keywords": ["RAG"],
        }

    def get_summary(self, paper_id, language, detail_level):
        if not self.built:
            return None
        return {
            "paper_id": paper_id,
            "language": language,
            "detail_level": detail_level,
            "summary_status": "ready",
            "summary": {"method": "Method", "results": "Results"},
        }

    def get_section_summaries(self, paper_id, language, detail_level):
        return [{"section": "Introduction", "summary": "Intro summary"}]

    def build_summary_cache(
        self,
        paper_id,
        language="en",
        detail_levels=None,
        force_rebuild=False,
        summary_generator=None,
    ):
        assert summary_generator is not None
        self.built = True
        return {
            "status": "ready",
            "paper_id": paper_id,
            "language": language,
            "detail_levels_built": detail_levels or ["medium"],
            "detail_levels_skipped": [],
        }


def test_retrieve_evidence_chunks_by_id_with_section(monkeypatch):
    monkeypatch.setattr(
        "rag.storage.paper_manager.get_chunks_for_paper_ids_readonly",
        lambda *args, **kwargs: [],
    )

    result = EvidenceChunkRetrievalTool(paper_manager=FakePaperManager()).execute(
        paper_ids="local_1",
        section="Introduction",
        top_k=2,
    )

    assert result.status == "success"
    assert "Section: Introduction" in result.result
    assert result.data["chunks"][0]["section_name"] == "introduction"


def test_retrieve_evidence_chunks_returns_structured_chunks():
    result = EvidenceChunkRetrievalTool(paper_manager=FakePaperManager()).execute(
        query="attention",
        top_k=2,
        max_chars=2000,
    )

    assert result.status == "success"
    assert result.data["chunks"][0]["chunk_id"] == "local_2:0"
    assert result.data["chunks"][0]["text"] == "Method text"


def test_summary_tools_read_cached_data_and_report_not_ready(monkeypatch):
    class FakeSummaryGenerator:
        model_name = "fake-summary-model"
        prompt_version = "fake-v1"

    monkeypatch.setattr(
        "rag.plugins.summary_model.get_summary_model_manager",
        lambda: FakeSummaryGenerator(),
    )
    db = FakeSummaryDB()

    outline = PaperOutlineTool(paper_db=db).execute("local_1")
    assert outline.status == "success"
    assert outline.data["sections"][0]["has_summary"] is False

    missing = PaperProfileTool(paper_db=db).execute("local_1")
    assert missing.status == "fail"
    assert missing.error_code == "SUMMARY_NOT_READY"

    build = BuildPaperSummaryTool(paper_db=db).execute("local_1", detail_levels=["medium"])
    finished = summary_job_manager.wait(build.data["job_id"], timeout=5)
    assert finished["status"] == "succeeded"

    profile = PaperProfileTool(paper_db=db).execute("local_1")
    assert profile.status == "success"
    assert profile.data["method"] == "Method"

    summary = PaperSummaryTool(paper_db=db).execute("local_1", detail_level="medium")
    assert summary.status == "success"
    assert summary.data["summary"]["results"] == "Results"


def test_build_summary_fails_when_summary_model_unavailable(monkeypatch):
    def fail_model_manager():
        raise RuntimeError("missing summary model")

    monkeypatch.setattr("rag.plugins.summary_model.get_summary_model_manager", fail_model_manager)

    result = BuildPaperSummaryTool(paper_db=FakeSummaryDB()).execute("local_1", detail_levels=["medium"])
    finished = summary_job_manager.wait(result.data["job_id"], timeout=5)

    assert finished["status"] == "failed"
    assert finished["error_code"] == "SUMMARY_MODEL_UNAVAILABLE"
    assert "missing summary model" in finished["result"]["result"]


def test_retrieve_evidence_chunks_rejects_invalid_local_id():
    result = EvidenceChunkRetrievalTool(paper_manager=FakePaperManager()).execute(
        paper_ids="../local_1",
        top_k=2,
    )

    assert result.status == "fail"
    assert result.error_code == "INVALID_LOCAL_ID"


def test_retrieve_evidence_chunks_rejects_long_query(monkeypatch):
    monkeypatch.setattr("rag.tools.paper_tools.conf.TOOL_MAX_QUERY_CHARS", 4)

    result = EvidenceChunkRetrievalTool(paper_manager=FakePaperManager()).execute(query="attention")

    assert result.status == "fail"
    assert result.error_code == "QUERY_TOO_LONG"


def test_tool_result_structured_error_payload():
    result = ToolResult.fail(
        "bad input",
        error_code="BAD_INPUT",
        suggestion="retry",
    )

    payload = result.to_payload()

    assert payload["status"] == "fail"
    assert payload["error_code"] == "BAD_INPUT"
    assert payload["recoverable"] is True
    assert payload["suggestion"] == "retry"
    assert result.to_mcp_text().startswith("[tool_error] ")


def test_execute_tool_safely_returns_missing_argument_error_code():
    result = execute_tool_safely(DbSearchTool(paper_db=FakePaperDB()), {})

    assert result.status == "fail"
    assert result.error_code == "MISSING_REQUIRED_ARGUMENT"
    assert result.request_id.startswith("tool_")
    assert isinstance(result.elapsed_ms, int)


def test_add_paper_submits_background_job(monkeypatch, tmp_path):
    class FakePaperManagerForIngest:
        def ingest_pdf(self, pdf_path, tags="", on_duplicate="skip"):
            assert pdf_path.endswith("demo.pdf")
            assert tags == "Agent-Added"
            assert on_duplicate == "skip"
            return True, "ingested"

    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))
    (tmp_path / "demo.pdf").write_text("pdf", encoding="utf-8")

    result = DbAddTool(paper_manager=FakePaperManagerForIngest()).execute("demo.pdf", build_summary=False)

    assert result.status == "success"
    assert result.data["job_id"].startswith("job_")

    finished = job_manager.wait(result.data["job_id"], timeout=5)
    assert finished["status"] == "succeeded"
    assert finished["request_id"].startswith("job_")
    assert isinstance(finished["elapsed_ms"], int)

    status = ToolJobStatusTool().execute(result.data["job_id"])
    assert status.status == "success"
    assert status.data["status"] == "succeeded"


def test_job_status_long_polls_running_job_with_arithmetic_backoff(monkeypatch):
    monkeypatch.setattr("rag.tools.paper_tools.conf.TOOL_JOB_STATUS_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("rag.tools.paper_tools.conf.TOOL_JOB_STATUS_WAIT_STEP_SECONDS", 0.02)
    monkeypatch.setattr("rag.tools.paper_tools.conf.TOOL_JOB_STATUS_WAIT_MAX_SECONDS", 0.05)
    started = threading.Event()
    release = threading.Event()

    def slow_job():
        started.set()
        release.wait(timeout=5)
        return ToolResult.success("done")

    job = job_manager.submit("unit", {}, slow_job)
    assert started.wait(timeout=2)

    first = ToolJobStatusTool().execute(job["job_id"])
    second = ToolJobStatusTool().execute(job["job_id"])

    assert first.status == "success"
    assert first.data["status"] == "running"
    assert first.data["status_checks"] == 1
    assert round(first.data["wait_seconds_applied"], 2) == 0.01
    assert round(first.data["next_wait_seconds"], 2) == 0.03
    assert second.status == "success"
    assert second.data["status"] == "running"
    assert second.data["status_checks"] == 2
    assert round(second.data["wait_seconds_applied"], 2) == 0.03
    assert round(second.data["next_wait_seconds"], 2) == 0.05

    release.set()
    finished = job_manager.wait(job["job_id"], timeout=5)
    assert finished["status"] == "succeeded"

    done = ToolJobStatusTool().execute(job["job_id"])
    assert done.data["status"] == "succeeded"
    assert done.data["status_checks"] == 3
    assert done.data["wait_seconds_applied"] == 0.0


def test_add_paper_auto_submits_summary_job(monkeypatch, tmp_path):
    submitted = []

    class FakePaperManagerForIngest:
        last_ingested_paper_id = ""

        def ingest_pdf(self, pdf_path, tags="", on_duplicate="skip"):
            self.last_ingested_paper_id = "local_1"
            return True, "ingested"

    def fake_submit_summary_job(*, paper_id, language, detail_levels, force_rebuild=False):
        submitted.append(
            {
                "paper_id": paper_id,
                "language": language,
                "detail_levels": detail_levels,
                "force_rebuild": force_rebuild,
            }
        )
        return {"job_id": "job_summary_1", "status": "running"}

    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))
    monkeypatch.setattr("rag.tools.paper_tools._submit_summary_build_job", fake_submit_summary_job)
    (tmp_path / "demo.pdf").write_text("pdf", encoding="utf-8")

    result = DbAddTool(paper_manager=FakePaperManagerForIngest()).execute(
        "demo.pdf",
        summary_language="zh",
        summary_detail_levels=["short", "medium"],
    )
    finished = job_manager.wait(result.data["job_id"], timeout=5)

    assert finished["status"] == "succeeded"
    assert finished["result"]["data"]["paper_id"] == "local_1"
    assert finished["result"]["data"]["summary_job"]["job_id"] == "job_summary_1"
    assert submitted == [
        {
            "paper_id": "local_1",
            "language": "zh",
            "detail_levels": ["short", "medium"],
            "force_rebuild": False,
        }
    ]


def test_add_paper_job_maps_hash_stage_failure(monkeypatch, tmp_path):
    class FakePaperManagerForHashFailure:
        def ingest_pdf(self, pdf_path, tags="", on_duplicate="skip"):
            return False, "Ingestion failed at stage=hash_pdf: OSError: [Errno 22] Invalid argument"

    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))
    (tmp_path / "demo.pdf").write_text("pdf", encoding="utf-8")

    result = DbAddTool(paper_manager=FakePaperManagerForHashFailure()).execute("demo.pdf", build_summary=False)
    finished = job_manager.wait(result.data["job_id"], timeout=5)

    assert finished["status"] == "failed"
    assert finished["error_code"] == "PDF_READ_FAILED"
    assert finished["result"]["suggestion"]


def test_add_paper_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))

    result = DbAddTool().execute("..\\demo.pdf")

    assert result.status == "fail"
    assert result.error_code == "PATH_TRAVERSAL_BLOCKED"


def test_import_directory_submits_background_job(monkeypatch, tmp_path):
    class FakePaperManagerForDirectoryImport:
        def ingest_directory(
            self,
            directory,
            *,
            recursive=False,
            on_duplicate="skip",
            max_files=200,
            dry_run=False,
            tags="",
        ):
            assert directory == str(tmp_path)
            assert recursive is True
            assert on_duplicate == "skip"
            assert max_files == 10
            assert dry_run is False
            assert tags == "Agent-Added"
            return {
                "status": "succeeded",
                "message": "Processed 2/2 PDF file(s): imported=1, skipped=0, failed=1.",
                "directory": directory,
                "recursive": recursive,
                "dry_run": dry_run,
                "max_files": max_files,
                "total_found": 2,
                "processed": 2,
                "imported": 1,
                "skipped": 0,
                "failed": 1,
                "limited": False,
                "results": [
                    {"filename": "ok.pdf", "status": "imported", "message": "ingested"},
                    {"filename": "bad.pdf", "status": "failed", "message": "parse failed"},
                ],
            }

    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))

    result = DbImportDirectoryTool(paper_manager=FakePaperManagerForDirectoryImport()).execute(
        recursive=True,
        max_files=10,
        build_summary=False,
    )

    assert result.status == "success"
    assert result.data["job_id"].startswith("job_")

    finished = job_manager.wait(result.data["job_id"], timeout=5)
    assert finished["status"] == "succeeded"
    assert finished["result"]["data"]["imported"] == 1
    assert finished["result"]["data"]["failed"] == 1
    assert "failed=1" in finished["result"]["result"]


def test_import_directory_auto_submits_summary_jobs(monkeypatch, tmp_path):
    submitted = []

    class FakePaperManagerForDirectoryImport:
        def ingest_directory(
            self,
            directory,
            *,
            recursive=False,
            on_duplicate="skip",
            max_files=200,
            dry_run=False,
            tags="",
        ):
            return {
                "status": "succeeded",
                "message": "Processed 2/2 PDF file(s): imported=1, skipped=1, failed=0.",
                "directory": directory,
                "recursive": recursive,
                "dry_run": dry_run,
                "max_files": max_files,
                "total_found": 2,
                "processed": 2,
                "imported": 1,
                "skipped": 1,
                "failed": 0,
                "limited": False,
                "results": [
                    {"filename": "ok.pdf", "status": "imported", "paper_id": "local_1", "message": "ingested"},
                    {"filename": "dup.pdf", "status": "skipped", "message": "Duplicate detected"},
                ],
            }

    def fake_submit_summary_job(*, paper_id, language, detail_levels, force_rebuild=False):
        submitted.append((paper_id, language, detail_levels, force_rebuild))
        return {"job_id": f"job_summary_{len(submitted)}", "status": "running", "paper_id": paper_id}

    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))
    monkeypatch.setattr("rag.tools.paper_tools._submit_summary_build_job", fake_submit_summary_job)

    result = DbImportDirectoryTool(paper_manager=FakePaperManagerForDirectoryImport()).execute(
        summary_language="en",
        summary_detail_levels=["long"],
    )
    finished = job_manager.wait(result.data["job_id"], timeout=5)

    assert finished["status"] == "succeeded"
    data = finished["result"]["data"]
    assert data["summary_jobs"][0]["job_id"] == "job_summary_1"
    assert data["results"][0]["summary_job_id"] == "job_summary_1"
    assert "summary_jobs_submitted=1" in finished["result"]["result"]
    assert submitted == [("local_1", "en", ["long"], False)]


def test_import_directory_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))

    result = DbImportDirectoryTool().execute(subdir="../outside")

    assert result.status == "fail"
    assert result.error_code == "PATH_TRAVERSAL_BLOCKED"


def test_import_directory_coerces_string_booleans(monkeypatch, tmp_path):
    class FakePaperManagerForStringBooleans:
        def ingest_directory(
            self,
            directory,
            *,
            recursive=False,
            on_duplicate="skip",
            max_files=200,
            dry_run=False,
            tags="",
        ):
            assert recursive is False
            assert dry_run is False
            return {
                "status": "succeeded",
                "message": "Processed 0/0 PDF file(s): imported=0, skipped=0, failed=0.",
                "directory": directory,
                "recursive": recursive,
                "dry_run": dry_run,
                "max_files": max_files,
                "total_found": 0,
                "processed": 0,
                "imported": 0,
                "skipped": 0,
                "failed": 0,
                "limited": False,
                "results": [],
            }

    monkeypatch.setattr("rag.tools.paper_tools.conf.PAPERS_DIR", str(tmp_path))

    result = DbImportDirectoryTool(paper_manager=FakePaperManagerForStringBooleans()).execute(
        recursive="false",
        dry_run="false",
        build_summary=False,
    )
    finished = job_manager.wait(result.data["job_id"], timeout=5)

    assert result.status == "success"
    assert finished["status"] == "succeeded"
    assert finished["result"]["data"]["recursive"] is False
    assert finished["result"]["data"]["dry_run"] is False


def test_search_local_database_rejects_long_query(monkeypatch):
    monkeypatch.setattr("rag.tools.paper_tools.conf.TOOL_MAX_QUERY_CHARS", 4)

    result = DbSearchTool(paper_db=FakePaperDB()).execute(query="attention")

    assert result.status == "fail"
    assert result.error_code == "QUERY_TOO_LONG"


def test_job_status_missing_job_has_error_code():
    result = ToolJobStatusTool().execute("job_missing")

    assert result.status == "fail"
    assert result.error_code == "JOB_NOT_FOUND"


def test_job_status_rejects_invalid_job_id():
    result = ToolJobStatusTool().execute("../job_1")

    assert result.status == "fail"
    assert result.error_code == "INVALID_JOB_ID"


def test_rag_health_check_returns_observability_data(monkeypatch):
    monkeypatch.setattr("rag.tools.paper_tools.conf.RAG_SUMMARY_API_BASE_URL", "")

    result = RagHealthCheckTool().execute()

    assert result.status == "success"
    assert "checks" in result.data
    assert "summary_model" in result.data
    assert "jobs" in result.data
    assert "summary_model_auto_download" in result.result
    assert "summary_model_vllm_installed" in result.result
    assert "jobs_total" in result.result


def test_retrieval_eval_tool_rejects_path_outside_project(tmp_path):
    result = RetrievalQualityEvalTool().execute(dataset_path=str(tmp_path / "eval.jsonl"))

    assert result.status == "fail"
    assert result.error_code == "PATH_TRAVERSAL_BLOCKED"
