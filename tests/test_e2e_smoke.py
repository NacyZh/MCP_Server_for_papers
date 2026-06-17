import subprocess
import sys
from pathlib import Path

import fitz

from rag.tools.jobs import job_manager
from rag.tools.paper_tools import DbAddTool, DbListTool, DbSearchTool, EvidenceChunkRetrievalTool


class FakeParser:
    def __init__(self, marker_device="auto"):
        self.marker_device = marker_device

    def process_paper(self, pdf_path):
        assert Path(pdf_path).is_file()
        return {
            "meta": {
                "title": "End To End RAG Smoke",
                "author": "ScholarAgent Tests",
                "doi": "10.0000/e2e-smoke",
            },
            "chunks": [
                {
                    "content": "The introduction describes local paper retrieval and MCP tooling.",
                    "section_name": "introduction",
                    "section_title": "Introduction",
                },
                {
                    "content": "The method combines SQLite metadata with vector chunks.",
                    "section_name": "method",
                    "section_title": "Method",
                },
            ],
        }


class FakeVectorDB:
    def __init__(self):
        self.chunks = []
        self.deleted = []

    def add_chunks(self, paper_id, chunks):
        self.chunks.extend(
            {**chunk, "paper_id": paper_id, "chunk_id": f"{paper_id}_chunk_{idx}"}
            for idx, chunk in enumerate(chunks)
        )

    def search(self, query_text, n_results=100, hybrid=True, embed_query=None, mode="", rerank=True):
        query = str(query_text or "").lower()
        matches = [
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "section_name": chunk["section_name"],
                "section_title": chunk["section_title"],
                "content": chunk["content"],
                "rerank_score": 1.0,
            }
            for chunk in self.chunks
            if query in chunk["content"].lower() or query in chunk["section_title"].lower()
        ]
        return matches[:n_results]

    def delete_paper(self, paper_id):
        self.deleted.append(paper_id)


def _write_minimal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "End To End RAG Smoke\n\nIntroduction\nLocal paper retrieval.")
    doc.save(path)
    doc.close()


def test_pdf_ingest_and_retrieval_toolchain_e2e(monkeypatch, tmp_path):
    import config as config_mod
    import rag.plugins.pdf_parser as pdf_parser
    import rag.storage.paper_manager as paper_manager
    import rag.storage.sqlite_store as sqlite_store

    db_dir = tmp_path / "db"
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "e2e.pdf"
    _write_minimal_pdf(pdf_path)

    for module_conf in (config_mod.conf, paper_manager.conf, sqlite_store.conf):
        monkeypatch.setattr(module_conf, "DB_DIR", str(db_dir))
        monkeypatch.setattr(module_conf, "PAPERS_DIR", str(papers_dir))
        monkeypatch.setattr(module_conf, "ENABLE_HYDE", False)

    monkeypatch.setattr(pdf_parser, "PaperParser", FakeParser)
    monkeypatch.setattr("rag.storage.vector_store.VectorDB", FakeVectorDB)

    add_result = DbAddTool().execute("e2e.pdf", build_summary=False)
    assert add_result.status == "success"

    job = job_manager.wait(add_result.data["job_id"], timeout=5)
    assert job["status"] == "succeeded"
    assert "Successfully ingested" in job["result"]["result"]

    list_result = DbListTool().execute()
    assert list_result.status == "success"
    assert "End To End RAG Smoke" in list_result.result
    assert "Introduction" in list_result.result

    search_result = DbSearchTool().execute(query="Smoke")
    assert search_result.status == "success"
    assert search_result.data["papers"][0]["title"] == "End To End RAG Smoke"

    paper_id = search_result.data["papers"][0]["id"]
    retrieve_result = EvidenceChunkRetrievalTool().execute(paper_ids=paper_id, section="Method", top_k=2)
    assert retrieve_result.status == "success"
    assert "SQLite metadata" in retrieve_result.result
    assert retrieve_result.data["chunks"][0]["section_name"] == "method"


def test_main_list_tools_subprocess_smoke():
    completed = subprocess.run(
        [sys.executable, "main.py", "--mode", "list-tools"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert "Registered MCP tools:" in completed.stdout
    assert "retrieve_evidence_chunks" in completed.stdout
    assert "retrieve_local_papers" not in completed.stdout
    assert "evaluate_retrieval_quality" in completed.stdout


def test_paper_parser_surfaces_marker_runtime_failure(monkeypatch, tmp_path):
    import pytest

    import rag.plugins.pdf_parser as pdf_parser

    pdf_path = tmp_path / "marker_runtime_failure.pdf"
    _write_minimal_pdf(pdf_path)

    class FailingConverter:
        def __call__(self, pdf_path):
            raise OSError(22, "Invalid argument")

    monkeypatch.setattr(pdf_parser, "create_model_dict", lambda device="cpu": {})
    monkeypatch.setattr(pdf_parser, "PdfConverter", lambda artifact_dict: FailingConverter())
    monkeypatch.setattr(pdf_parser, "text_from_rendered", lambda rendered: ("", None, None))

    parser = pdf_parser.PaperParser(marker_device="cpu")

    with pytest.raises(OSError):
        parser.process_paper(str(pdf_path))


def test_paper_parser_surfaces_marker_initialization_failure(monkeypatch, tmp_path):
    import pytest

    import rag.plugins.pdf_parser as pdf_parser

    pdf_path = tmp_path / "marker_init_failure.pdf"
    _write_minimal_pdf(pdf_path)

    def fail_model_dict(device="cpu"):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(pdf_parser, "create_model_dict", fail_model_dict)
    monkeypatch.setattr(pdf_parser, "PdfConverter", lambda artifact_dict: object())
    monkeypatch.setattr(pdf_parser, "text_from_rendered", lambda rendered: ("", None, None))

    parser = pdf_parser.PaperParser(marker_device="cpu")

    with pytest.raises(OSError):
        parser.process_paper(str(pdf_path))
