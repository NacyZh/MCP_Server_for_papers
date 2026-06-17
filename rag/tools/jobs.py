"""In-process job manager for long-running MCP tools."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict

from rag.core.logging import get_logger
from rag.core.observability import log_event, new_request_id
from rag.tools.base import ToolResult

logger = get_logger(__name__)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobManager:
    def __init__(self, max_workers: int = 2, *, name: str = "default", thread_name_prefix: str = "rag-job"):
        self.name = str(name or "default")
        self.max_workers = max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=thread_name_prefix)
        self._lock = threading.Lock()
        self._jobs: Dict[str, dict] = {}

    def submit(self, job_type: str, params: dict, fn: Callable[[], ToolResult]) -> dict:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        request_id = new_request_id("job")
        record = {
            "job_id": job_id,
            "request_id": request_id,
            "job_type": job_type,
            "queue": self.name,
            "status": JOB_STATUS_QUEUED,
            "status_checks": 0,
            "params": params,
            "created_at": _now_iso(),
            "started_at": "",
            "finished_at": "",
            "result": None,
            "error_code": "",
            "recoverable": True,
            "suggestion": "",
            "elapsed_ms": None,
        }
        with self._lock:
            self._jobs[job_id] = record
        log_event("job_submitted", request_id=request_id, job_id=job_id, job_type=job_type)
        self._executor.submit(self._run, job_id, fn)
        return self.get(job_id) or record

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            record = self._jobs.get(str(job_id or ""))
            return dict(record) if record else None

    def observe_status(self, job_id: str) -> dict | None:
        with self._lock:
            record = self._jobs.get(str(job_id or ""))
            if record is None:
                return None
            record["status_checks"] = int(record.get("status_checks") or 0) + 1
            return dict(record)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(job_id)
            if record is None or record["status"] in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED}:
                return record
            time.sleep(0.02)
        return self.get(job_id)

    def stats(self) -> dict:
        with self._lock:
            records = list(self._jobs.values())
        return {
            "queue": self.name,
            "max_workers": self.max_workers,
            "total": len(records),
            "queued": sum(1 for item in records if item["status"] == JOB_STATUS_QUEUED),
            "running": sum(1 for item in records if item["status"] == JOB_STATUS_RUNNING),
            "succeeded": sum(1 for item in records if item["status"] == JOB_STATUS_SUCCEEDED),
            "failed": sum(1 for item in records if item["status"] == JOB_STATUS_FAILED),
        }

    def _run(self, job_id: str, fn: Callable[[], ToolResult]) -> None:
        started = time.perf_counter()
        with self._lock:
            self._jobs[job_id]["status"] = JOB_STATUS_RUNNING
            self._jobs[job_id]["started_at"] = _now_iso()
            request_id = self._jobs[job_id]["request_id"]
            job_type = self._jobs[job_id]["job_type"]
        log_event("job_started", request_id=request_id, job_id=job_id, job_type=job_type)
        try:
            result = fn()
            status = JOB_STATUS_SUCCEEDED if result.status == "success" else JOB_STATUS_FAILED
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            result.request_id = result.request_id or request_id
            result.elapsed_ms = elapsed_ms
            with self._lock:
                record = self._jobs[job_id]
                record["status"] = status
                record["finished_at"] = _now_iso()
                record["result"] = result.to_payload()
                record["error_code"] = result.error_code
                record["recoverable"] = result.recoverable
                record["suggestion"] = result.suggestion
                record["elapsed_ms"] = elapsed_ms
            log_event(
                "job_finished",
                request_id=request_id,
                job_id=job_id,
                job_type=job_type,
                status=status,
                error_code=result.error_code,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.exception("[job] unhandled job failure job_id=%s", job_id)
            with self._lock:
                record = self._jobs[job_id]
                record["status"] = JOB_STATUS_FAILED
                record["finished_at"] = _now_iso()
                record["error_code"] = "UNHANDLED_JOB_EXCEPTION"
                record["recoverable"] = False
                record["suggestion"] = "Check server logs for the traceback."
                record["elapsed_ms"] = elapsed_ms
                record["result"] = {
                    "status": "fail",
                    "result": f"Job failed: {exc}",
                    "error_code": "UNHANDLED_JOB_EXCEPTION",
                    "recoverable": False,
                    "suggestion": "Check server logs for the traceback.",
                    "request_id": record["request_id"],
                    "elapsed_ms": elapsed_ms,
                    "traceback": traceback.format_exc(),
                }
            log_event(
                "job_failed",
                request_id=request_id,
                job_id=job_id,
                job_type=job_type,
                error_code="UNHANDLED_JOB_EXCEPTION",
                elapsed_ms=elapsed_ms,
            )


job_manager = JobManager()
