from rag.storage.paper_manager import PaperManager


class FakeBatchPaperManager(PaperManager):
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}

    def ingest_pdf(self, pdf_path, paper_id=None, tags="", on_duplicate="skip"):
        self.calls.append((pdf_path, tags, on_duplicate))
        name = pdf_path.rsplit("/", 1)[-1]
        return self.outcomes.get(name, (True, f"ingested {name}"))


def test_ingest_directory_dry_run_lists_pdfs_without_importing(tmp_path):
    (tmp_path / "b.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "a.PDF").write_text("pdf", encoding="utf-8")
    (tmp_path / "note.txt").write_text("not pdf", encoding="utf-8")
    manager = FakeBatchPaperManager()

    report = manager.ingest_directory(tmp_path, dry_run=True)

    assert report["status"] == "succeeded"
    assert report["total_found"] == 2
    assert report["processed"] == 0
    assert report["skipped"] == 2
    assert [item["relative_path"] for item in report["results"]] == ["a.PDF", "b.pdf"]
    assert manager.calls == []


def test_ingest_directory_continues_after_file_failure(tmp_path):
    for name in ("ok.pdf", "dup.pdf", "bad.pdf"):
        (tmp_path / name).write_text("pdf", encoding="utf-8")
    manager = FakeBatchPaperManager(
        {
            "dup.pdf": (False, "Duplicate detected by content_hash"),
            "bad.pdf": (False, "Ingestion failed at stage=parse_pdf: RuntimeError: broken"),
        }
    )

    report = manager.ingest_directory(tmp_path, on_duplicate="skip", tags="Batch")

    assert report["status"] == "succeeded"
    assert report["processed"] == 3
    assert report["imported"] == 1
    assert report["skipped"] == 1
    assert report["failed"] == 1
    assert [item["status"] for item in report["results"]] == ["failed", "skipped", "imported"]
    assert all(call[1] == "Batch" for call in manager.calls)
    assert all(call[2] == "skip" for call in manager.calls)


def test_ingest_directory_honors_recursive_and_max_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "root.pdf").write_text("pdf", encoding="utf-8")
    (nested / "child.pdf").write_text("pdf", encoding="utf-8")
    manager = FakeBatchPaperManager()

    non_recursive = manager.ingest_directory(tmp_path)
    recursive_limited = manager.ingest_directory(tmp_path, recursive=True, max_files=1)

    assert non_recursive["total_found"] == 1
    assert recursive_limited["total_found"] == 2
    assert recursive_limited["processed"] == 1
    assert recursive_limited["limited"] is True
