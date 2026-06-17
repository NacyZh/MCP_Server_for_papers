import sys
import time
import types
from pathlib import Path

import pytest

from rag.storage import model_manager


def write_minimal_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")


class FakeHealthResponse:
    def __init__(self, status_code=200, text='{"data":[]}'):
        self.status_code = status_code
        self.text = text


def test_existing_complete_model_does_not_download(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    write_minimal_model(model_path)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not download"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    result = model_manager.ensure_local_model(str(model_path), "repo/model", "main", "test model")

    assert result == str(model_path)


def test_offline_missing_model_fails_without_download(monkeypatch, tmp_path):
    downloads = []
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *args, **kwargs: downloads.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(model_manager.conf, "BGE_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "BGE_OFFLINE_MODE", True)

    with pytest.raises(model_manager.ModelUnavailableError):
        model_manager.ensure_local_model(str(tmp_path / "missing"), "repo/model", "main", "test model")

    assert downloads == []


def test_failed_download_cleans_temp_dir(monkeypatch, tmp_path):
    def fail_download(repo_id, revision, local_dir, local_dir_use_symlinks=False):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "partial.bin").write_text("partial", encoding="utf-8")
        raise RuntimeError("network failed")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fail_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(model_manager.conf, "BGE_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "BGE_OFFLINE_MODE", False)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_TIMEOUT_SEC", 2)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_STALE_SEC", 60)

    model_path = tmp_path / "model"
    with pytest.raises(model_manager.ModelUnavailableError):
        model_manager.ensure_local_model(str(model_path), "repo/model", "main", "test model")

    assert not model_path.exists()
    assert list(tmp_path.glob(".download-model-*")) == []


def test_interrupted_download_cleans_temp_dir(monkeypatch, tmp_path):
    def interrupt_download(repo_id, revision, local_dir, local_dir_use_symlinks=False):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "partial.bin").write_text("partial", encoding="utf-8")
        raise KeyboardInterrupt

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = interrupt_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(model_manager.conf, "BGE_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "BGE_OFFLINE_MODE", False)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_TIMEOUT_SEC", 2)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_STALE_SEC", 60)

    model_path = tmp_path / "model"
    with pytest.raises(KeyboardInterrupt):
        model_manager.ensure_local_model(str(model_path), "repo/model", "main", "test model")

    assert not model_path.exists()
    assert list(tmp_path.glob(".download-model-*")) == []


def test_model_lock_timeout(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    lock_path = Path(f"{model_path}{model_manager.MODEL_LOCK_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("locked", encoding="utf-8")

    monkeypatch.setattr(model_manager.conf, "BGE_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "BGE_OFFLINE_MODE", False)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_STALE_SEC", 3600)

    with pytest.raises(model_manager.ModelLockTimeoutError):
        model_manager.ensure_local_model(str(model_path), "repo/model", "main", "test model")


def test_stale_model_lock_is_removed(monkeypatch, tmp_path):
    downloads = []

    def fake_download(repo_id, revision, local_dir, local_dir_use_symlinks=False):
        downloads.append(repo_id)
        write_minimal_model(Path(local_dir))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    model_path = tmp_path / "model"
    lock_path = Path(f"{model_path}{model_manager.MODEL_LOCK_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale", encoding="utf-8")
    old = time.time() - 7200
    Path(lock_path).touch()
    import os

    os.utime(lock_path, (old, old))

    monkeypatch.setattr(model_manager.conf, "BGE_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "BGE_OFFLINE_MODE", False)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_TIMEOUT_SEC", 2)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_STALE_SEC", 1)

    assert model_manager.ensure_local_model(str(model_path), "repo/model", "main", "test model") == str(model_path)
    assert downloads == ["repo/model"]
    assert not lock_path.exists()


def test_summary_model_offline_missing_fails_without_download(monkeypatch, tmp_path):
    downloads = []
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *args, **kwargs: downloads.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(tmp_path / "missing-summary"))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REPO", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_NAME", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REVISION", "main")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_OFFLINE_MODE", True)

    with pytest.raises(model_manager.ModelUnavailableError):
        model_manager.ensure_summary_model()

    assert downloads == []


def test_summary_model_download_uses_configured_repo(monkeypatch, tmp_path):
    downloads = []

    def fake_download(repo_id, revision, local_dir, local_dir_use_symlinks=False):
        downloads.append((repo_id, revision))
        write_minimal_model(Path(local_dir))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(tmp_path / "summary"))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REPO", "Qwen/Qwen3-4B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_NAME", "Qwen/Qwen3-4B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REVISION", "main")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_OFFLINE_MODE", False)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_TIMEOUT_SEC", 2)
    monkeypatch.setattr(model_manager.conf, "BGE_MODEL_LOCK_STALE_SEC", 60)

    assert model_manager.ensure_summary_model() == str(tmp_path / "summary")
    assert downloads == [("Qwen/Qwen3-4B", "main")]
    assert model_manager.get_summary_model_status()["complete"] is True


def test_summary_model_status_reports_download_and_runtime_observability(monkeypatch, tmp_path):
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(tmp_path / "missing-summary"))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REPO", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_NAME", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REVISION", "main")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_OFFLINE_MODE", False)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_BACKEND", "vllm")
    monkeypatch.setattr(
        model_manager,
        "find_spec",
        lambda name: object() if name == "huggingface_hub" else None,
    )

    status = model_manager.get_summary_model_status()

    assert status["complete"] is False
    assert status["auto_download"] is True
    assert status["offline_mode"] is False
    assert status["huggingface_hub_installed"] is True
    assert status["vllm_installed"] is False
    assert status["model_download_available"] is True
    assert status["download_would_start"] is False
    assert status["generation_ready"] is False
    assert status["generation_blocker"] == "vllm_not_installed"
    assert status["next_action"] == "install the summary-vllm extra in the active Python environment"


def test_summary_model_status_api_backend_does_not_require_local_model(monkeypatch, tmp_path):
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(tmp_path / "missing-summary"))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_NAME", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_BACKEND", "api")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_API_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_API_MODEL", "summary-api-model")
    monkeypatch.setattr(
        model_manager,
        "_get_with_hard_timeout",
        lambda *args, **kwargs: FakeHealthResponse(200),
    )

    status = model_manager.get_summary_model_status()

    assert status["complete"] is False
    assert status["backend"] == "api"
    assert status["api_base_url"] == "http://127.0.0.1:8001/v1"
    assert status["api_model"] == "summary-api-model"
    assert status["model_download_available"] is False
    assert status["download_blocker"] == "api_backend_does_not_use_local_model"
    assert status["api_ping_attempted"] is True
    assert status["api_ping_ok"] is True
    assert status["api_ping_status_code"] == 200
    assert status["generation_ready"] is True
    assert status["generation_blocker"] == ""


def test_summary_model_status_api_backend_reports_ping_failure(monkeypatch, tmp_path):
    def fail_ping(*args, **kwargs):
        raise TimeoutError("health timeout")

    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(tmp_path / "missing-summary"))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_BACKEND", "api")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_API_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_API_MODEL", "summary-api-model")
    monkeypatch.setattr(model_manager, "_get_with_hard_timeout", fail_ping)

    status = model_manager.get_summary_model_status()

    assert status["backend"] == "api"
    assert status["api_ping_attempted"] is True
    assert status["api_ping_ok"] is False
    assert "health timeout" in status["api_ping_error"]
    assert status["generation_ready"] is False
    assert status["generation_blocker"] == "summary_api_unreachable"
    assert status["next_action"] == "start the summary API service or fix RAG_SUMMARY_API_BASE_URL/RAG_SUMMARY_API_KEY"
