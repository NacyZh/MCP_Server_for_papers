"""Production-oriented local model management for retrieval models."""

from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterator

from config import conf
from rag.core.logging import get_logger

logger = get_logger(__name__)

MODEL_READY_FILENAME = ".scholaragent_model_ready.json"
MODEL_LOCK_SUFFIX = ".lock"

_REQUIRED_MODEL_FILES = (
    "config.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
    "model.safetensors",
)


class ModelManagerError(RuntimeError):
    """Base class for local model manager errors."""


class ModelUnavailableError(FileNotFoundError, ModelManagerError):
    """Raised when a required model cannot be used locally."""


class ModelLockTimeoutError(TimeoutError, ModelManagerError):
    """Raised when another process holds a model download lock too long."""


def ensure_bge_embedding_model() -> str:
    return ensure_local_model(
        model_name_or_path=conf.BGE_M3_MODEL_PATH,
        repo_id=conf.BGE_M3_MODEL_REPO,
        revision=conf.BGE_M3_MODEL_REVISION,
        label="BGE-M3 embedding model",
    )


def ensure_bge_reranker_model() -> str:
    return ensure_local_model(
        model_name_or_path=conf.BGE_RERANKER_MODEL_PATH,
        repo_id=conf.BGE_RERANKER_MODEL_REPO,
        revision=conf.BGE_RERANKER_MODEL_REVISION,
        label="BGE reranker model",
    )


def ensure_summary_model() -> str:
    return ensure_local_model(
        model_name_or_path=conf.RAG_SUMMARY_MODEL_PATH,
        repo_id=conf.RAG_SUMMARY_MODEL_REPO or conf.RAG_SUMMARY_MODEL_NAME,
        revision=conf.RAG_SUMMARY_MODEL_REVISION,
        label="summary generation model",
        auto_download=conf.RAG_SUMMARY_AUTO_DOWNLOAD,
        offline_mode=conf.RAG_SUMMARY_OFFLINE_MODE,
    )


def get_summary_model_status() -> dict:
    status = get_local_model_status(
        model_name_or_path=conf.RAG_SUMMARY_MODEL_PATH,
        repo_id=conf.RAG_SUMMARY_MODEL_REPO or conf.RAG_SUMMARY_MODEL_NAME,
        revision=conf.RAG_SUMMARY_MODEL_REVISION,
        label="summary generation model",
    )
    path = Path(str(status.get("path") or conf.RAG_SUMMARY_MODEL_PATH)).expanduser()
    lock_path = Path(f"{path}{MODEL_LOCK_SUFFIX}")
    hf_installed = _module_available("huggingface_hub")
    vllm_installed = _module_available("vllm")
    backend = str(conf.RAG_SUMMARY_BACKEND or "").strip().lower()
    complete = bool(status.get("complete"))
    path_like = bool(status.get("is_path_like"))
    auto_download = bool(conf.RAG_SUMMARY_AUTO_DOWNLOAD)
    offline_mode = bool(conf.RAG_SUMMARY_OFFLINE_MODE)
    runtime_status = _get_summary_runtime_status(path if path_like else None, complete) if backend == "vllm" else {}
    api_status = _ping_summary_api() if backend == "api" else {}

    model_download_available = bool(
        backend == "vllm"
        and path_like
        and not complete
        and auto_download
        and not offline_mode
        and hf_installed
    )
    download_blocker = ""
    if backend == "api":
        download_blocker = "api_backend_does_not_use_local_model"
    elif complete:
        download_blocker = "model_already_complete"
    elif not path_like:
        download_blocker = "model_path_is_not_local_path"
    elif offline_mode:
        download_blocker = "offline_mode_enabled"
    elif not auto_download:
        download_blocker = "auto_download_disabled"
    elif not hf_installed:
        download_blocker = "huggingface_hub_not_installed"

    runtime_blocker = ""
    if backend == "api" and not str(conf.RAG_SUMMARY_API_BASE_URL or "").strip():
        runtime_blocker = "summary_api_base_url_missing"
    elif backend == "api" and not bool(api_status.get("api_ping_ok")):
        runtime_blocker = "summary_api_unreachable"
    elif backend == "vllm" and not vllm_installed:
        runtime_blocker = "vllm_not_installed"
    elif backend == "vllm":
        runtime_blocker = str(runtime_status.get("runtime_blocker") or "")
    generation_blocker = runtime_blocker
    if backend == "vllm" and not generation_blocker and not complete:
        generation_blocker = "summary_model_missing"
    download_would_start = bool(model_download_available and not runtime_blocker)

    next_action = "ready"
    if runtime_blocker == "summary_api_base_url_missing":
        next_action = "set RAG_SUMMARY_API_BASE_URL to an OpenAI-compatible /v1 endpoint"
    elif runtime_blocker == "summary_api_unreachable":
        next_action = "start the summary API service or fix RAG_SUMMARY_API_BASE_URL/RAG_SUMMARY_API_KEY"
    elif runtime_blocker == "vllm_not_installed":
        next_action = "install the summary-vllm extra in the active Python environment"
    elif download_would_start:
        next_action = "call build_paper_summary to download the configured summary model"
    elif backend == "vllm" and download_blocker and download_blocker != "model_already_complete":
        next_action = "fix the download blocker or place a complete model at RAG_SUMMARY_MODEL_PATH"
    elif generation_blocker == "summary_model_missing":
        next_action = "place a complete model at RAG_SUMMARY_MODEL_PATH"

    status.update(
        {
            **runtime_status,
            **api_status,
            "backend": backend,
            "model_name": conf.RAG_SUMMARY_MODEL_NAME,
            "api_base_url": conf.RAG_SUMMARY_API_BASE_URL,
            "api_model": conf.RAG_SUMMARY_API_MODEL or conf.RAG_SUMMARY_MODEL_NAME,
            "api_timeout_seconds": conf.RAG_SUMMARY_API_TIMEOUT_SECONDS,
            "api_max_retries": conf.RAG_SUMMARY_API_MAX_RETRIES,
            "api_health_timeout_seconds": conf.RAG_SUMMARY_API_HEALTH_TIMEOUT_SECONDS,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "auto_download": auto_download,
            "offline_mode": offline_mode,
            "device": conf.RAG_SUMMARY_DEVICE,
            "dtype": conf.RAG_SUMMARY_DTYPE,
            "max_input_tokens": conf.RAG_SUMMARY_MAX_INPUT_TOKENS,
            "gpu_memory_utilization": conf.RAG_SUMMARY_GPU_MEMORY_UTILIZATION,
            "cpu_offload_gb": conf.RAG_SUMMARY_CPU_OFFLOAD_GB,
            "enforce_eager": conf.RAG_SUMMARY_ENFORCE_EAGER,
            "huggingface_hub_installed": hf_installed,
            "vllm_installed": vllm_installed,
            "download_trigger": "build_paper_summary_generation",
            "model_download_available": model_download_available,
            "lock_path": str(lock_path),
            "lock_exists": lock_path.exists(),
            "download_would_start": download_would_start,
            "download_blocker": download_blocker,
            "runtime_blocker": runtime_blocker,
            "generation_ready": (backend == "api" and not generation_blocker)
            or (backend == "vllm" and complete and not generation_blocker),
            "generation_blocker": generation_blocker,
            "next_action": next_action,
        }
    )
    return status


def _get_summary_runtime_status(path: Path | None, complete: bool) -> dict:
    status: dict[str, object] = {
        "model_weight_bytes": 0,
        "gpu_available": None,
        "gpu_name": "",
        "gpu_memory_free_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_memory_budget_bytes": None,
        "cpu_offload_available": None,
        "runtime_blocker": "",
    }
    if path is not None and complete:
        status["model_weight_bytes"] = _model_weight_size_bytes(path)

    if str(conf.RAG_SUMMARY_DEVICE or "").lower() != "cuda":
        return status

    cpu_offload_gb = float(conf.RAG_SUMMARY_CPU_OFFLOAD_GB)
    is_wsl = "microsoft" in platform.release().lower() or "wsl" in platform.release().lower()
    status["cpu_offload_available"] = not is_wsl
    if cpu_offload_gb > 0 and is_wsl:
        status["runtime_blocker"] = "summary_cpu_offload_unavailable_on_wsl"
        return status

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        status["gpu_available"] = cuda_available
        if not cuda_available:
            status["runtime_blocker"] = "cuda_not_available"
            return status
        free_bytes_raw, total_bytes_raw = torch.cuda.mem_get_info()
        free_bytes = int(free_bytes_raw)
        total_bytes = int(total_bytes_raw)
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_memory_free_bytes"] = free_bytes
        status["gpu_memory_total_bytes"] = total_bytes
    except Exception as exc:
        status["runtime_blocker"] = f"cuda_memory_check_failed:{exc}"
        return status

    if not complete:
        return status

    weight_bytes = _model_weight_size_bytes(path) if path is not None else 0
    gpu_budget_bytes = int(total_bytes * float(conf.RAG_SUMMARY_GPU_MEMORY_UTILIZATION))
    offload_bytes = int(cpu_offload_gb * 1024**3)
    effective_budget_bytes = gpu_budget_bytes + offload_bytes
    status["gpu_memory_budget_bytes"] = gpu_budget_bytes

    if free_bytes and free_bytes < gpu_budget_bytes:
        status["runtime_blocker"] = "summary_gpu_free_memory_below_utilization_target"
    elif weight_bytes and weight_bytes + 256 * 1024**2 > effective_budget_bytes:
        status["runtime_blocker"] = "summary_model_gpu_memory_insufficient"
    return status


def _ping_summary_api() -> dict:
    base_url = str(conf.RAG_SUMMARY_API_BASE_URL or "").strip()
    status: dict[str, object] = {
        "api_ping_attempted": False,
        "api_ping_ok": False,
        "api_ping_url": "",
        "api_ping_status_code": None,
        "api_ping_latency_ms": None,
        "api_ping_error": "",
    }
    if not base_url:
        return status

    url = _summary_api_models_endpoint(base_url)
    status["api_ping_attempted"] = True
    status["api_ping_url"] = url
    started = time.perf_counter()
    try:
        response = _get_with_hard_timeout(
            url,
            headers=_summary_api_headers(),
            timeout_seconds=float(conf.RAG_SUMMARY_API_HEALTH_TIMEOUT_SECONDS),
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        status["api_ping_status_code"] = status_code
        status["api_ping_latency_ms"] = int((time.perf_counter() - started) * 1000)
        if 200 <= status_code < 500:
            status["api_ping_ok"] = True
        else:
            text = str(getattr(response, "text", "") or "")
            status["api_ping_error"] = f"HTTP {status['api_ping_status_code']}: {text[:200]}"
    except Exception as exc:
        status["api_ping_latency_ms"] = int((time.perf_counter() - started) * 1000)
        status["api_ping_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _summary_api_models_endpoint(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if value.endswith("/models"):
        return value
    if value.endswith("/chat/completions"):
        return value.rsplit("/chat/completions", 1)[0] + "/models"
    return f"{value}/models"


def _summary_api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = conf.resolve_api_key(
        api_key=conf.RAG_SUMMARY_API_KEY,
        base_url=conf.RAG_SUMMARY_API_BASE_URL,
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_with_hard_timeout(url: str, *, headers: dict[str, str], timeout_seconds: float) -> Any:
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def run_request() -> None:
        try:
            import requests

            connect_timeout = min(2.0, max(0.1, float(timeout_seconds)))
            response = requests.get(
                url,
                headers=headers,
                timeout=(connect_timeout, max(0.1, float(timeout_seconds))),
            )
            result_queue.put(("response", response))
        except Exception as exc:
            result_queue.put(("error", exc))

    worker = threading.Thread(target=run_request, name="summary-api-health", daemon=True)
    worker.start()
    try:
        kind, value = result_queue.get(timeout=max(0.1, float(timeout_seconds)))
    except queue.Empty as exc:
        raise TimeoutError(f"summary API health ping exceeded timeout: {timeout_seconds:.1f}s") from exc
    if kind == "error":
        raise value
    return value


def _model_weight_size_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    patterns = ("*.safetensors", "*.bin", "*.pt")
    total = 0
    for pattern in patterns:
        for item in path.glob(pattern):
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def get_local_model_status(model_name_or_path: str, repo_id: str, revision: str, label: str) -> dict:
    raw = str(model_name_or_path or "").strip()
    status = {
        "label": label,
        "path": raw,
        "repo_id": repo_id,
        "revision": revision,
        "is_path_like": _is_path_like(raw) if raw else False,
        "exists": False,
        "complete": False,
        "ready_marker": False,
    }
    if not raw or not status["is_path_like"]:
        return status
    path = Path(raw).expanduser()
    status["path"] = str(path)
    status["exists"] = path.exists()
    status["ready_marker"] = _ready_marker_path(path).exists()
    status["complete"] = _is_complete_local_model(path)
    return status


def ensure_local_model(
    model_name_or_path: str,
    repo_id: str,
    revision: str,
    label: str,
    *,
    auto_download: bool | None = None,
    offline_mode: bool | None = None,
) -> str:
    """Return a usable model path or download it under a process-safe lock."""

    raw = str(model_name_or_path or "").strip()
    if not raw:
        raise ModelUnavailableError(f"{label} path is empty.")

    if not _is_path_like(raw):
        return raw

    path = Path(raw).expanduser()
    if _is_complete_local_model(path):
        return str(path)

    if path.exists() and not _has_required_model_files(path):
        logger.warning("[model] incomplete local model detected label=%s path=%s", label, path)

    if auto_download is None:
        auto_download = conf.BGE_AUTO_DOWNLOAD
    if offline_mode is None:
        offline_mode = conf.BGE_OFFLINE_MODE

    if offline_mode or not auto_download:
        raise ModelUnavailableError(
            f"{label} is not available at {path}. "
            "Enable automatic download with offline mode disabled, or place a complete model at this path."
        )

    with _model_download_lock(path, label):
        if _is_complete_local_model(path):
            return str(path)
        _download_model_atomic(path=path, repo_id=repo_id, revision=revision, label=label)
        if not _is_complete_local_model(path):
            raise ModelUnavailableError(f"{label} download finished but model validation failed: {path}")
        return str(path)


def _is_path_like(value: str) -> bool:
    return Path(value).is_absolute() or os.sep in value or (os.altsep is not None and os.altsep in value)


def _ready_marker_path(path: Path) -> Path:
    return path / MODEL_READY_FILENAME


def _has_required_model_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in _REQUIRED_MODEL_FILES)


def _is_complete_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    marker = _ready_marker_path(path)
    if marker.exists():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            return bool(data.get("complete")) and _has_required_model_files(path)
        except (OSError, json.JSONDecodeError):
            return False
    return _has_required_model_files(path)


@contextmanager
def _model_download_lock(path: Path, label: str) -> Iterator[None]:
    lock_path = Path(f"{path}{MODEL_LOCK_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(conf.BGE_MODEL_LOCK_TIMEOUT_SEC)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps({"pid": os.getpid(), "time": time.time(), "label": label}).encode("utf-8"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            if _lock_is_stale(lock_path):
                try:
                    lock_path.unlink()
                    logger.warning("[model] removed stale download lock path=%s", lock_path)
                    continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise ModelLockTimeoutError(f"Timed out waiting for {label} download lock: {lock_path}")
            time.sleep(1.0)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _lock_is_stale(lock_path: Path) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > float(conf.BGE_MODEL_LOCK_STALE_SEC)


def _download_model_atomic(path: Path, repo_id: str, revision: str, label: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelUnavailableError(
            f"{label} is not available at {path}, and huggingface_hub is not installed."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".download-{path.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)

    logger.info("[model] downloading label=%s repo=%s revision=%s target=%s", label, repo_id, revision, path)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision or None,
            local_dir=str(tmp_path),
            local_dir_use_symlinks=False,
        )
        if not _has_required_model_files(tmp_path):
            raise ModelUnavailableError(f"{label} download from {repo_id!r} did not contain required model files.")

        marker = {
            "complete": True,
            "repo_id": repo_id,
            "revision": revision or "",
            "label": label,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _ready_marker_path(tmp_path).write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")

        backup_path = None
        if path.exists():
            backup_path = path.parent / f".backup-{path.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            path.rename(backup_path)
        tmp_path.rename(path)
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
    except BaseException as exc:
        shutil.rmtree(tmp_path, ignore_errors=True)
        if isinstance(exc, Exception):
            raise ModelUnavailableError(f"{label} automatic download failed from {repo_id!r}: {exc}") from exc
        raise
