import threading

from rag.tools.base import ToolResult
from rag.tools.jobs import JOB_STATUS_FAILED, JOB_STATUS_SUCCEEDED, JobManager


def test_job_manager_records_success_metadata():
    manager = JobManager(max_workers=1)

    job = manager.submit("unit", {"x": 1}, lambda: ToolResult.success("done"))
    finished = manager.wait(job["job_id"], timeout=5)

    assert finished["status"] == JOB_STATUS_SUCCEEDED
    assert finished["request_id"].startswith("job_")
    assert isinstance(finished["elapsed_ms"], int)
    assert finished["result"]["request_id"] == finished["request_id"]
    assert finished["result"]["elapsed_ms"] == finished["elapsed_ms"]


def test_job_manager_records_failure_metadata():
    manager = JobManager(max_workers=1)

    job = manager.submit(
        "unit",
        {},
        lambda: ToolResult.fail("bad", error_code="UNIT_FAILED", suggestion="fix it"),
    )
    finished = manager.wait(job["job_id"], timeout=5)

    assert finished["status"] == JOB_STATUS_FAILED
    assert finished["error_code"] == "UNIT_FAILED"
    assert finished["result"]["error_code"] == "UNIT_FAILED"
    assert finished["result"]["request_id"] == finished["request_id"]


def test_job_manager_records_unhandled_exception():
    manager = JobManager(max_workers=1)

    def boom():
        raise RuntimeError("boom")

    job = manager.submit("unit", {}, boom)
    finished = manager.wait(job["job_id"], timeout=5)

    assert finished["status"] == JOB_STATUS_FAILED
    assert finished["error_code"] == "UNHANDLED_JOB_EXCEPTION"
    assert finished["result"]["recoverable"] is False


def test_job_manager_preserves_queued_status_when_worker_is_busy():
    manager = JobManager(max_workers=1, name="limited")
    started = threading.Event()
    release = threading.Event()

    def wait_until_released():
        started.set()
        release.wait(timeout=5)
        return ToolResult.success("released")

    first = manager.submit("slow", {}, wait_until_released)
    assert started.wait(timeout=2)
    second = manager.submit("fast", {}, lambda: ToolResult.success("done"))

    stats = manager.stats()
    assert stats["queue"] == "limited"
    assert stats["max_workers"] == 1
    assert stats["running"] == 1
    assert stats["queued"] == 1

    queued = manager.get(second["job_id"])
    assert queued["status"] == "queued"

    release.set()
    assert manager.wait(first["job_id"], timeout=5)["status"] == JOB_STATUS_SUCCEEDED
    assert manager.wait(second["job_id"], timeout=5)["status"] == JOB_STATUS_SUCCEEDED
