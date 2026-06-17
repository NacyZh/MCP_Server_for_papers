import builtins
import json
import sys
import time
import types
from pathlib import Path

import pytest

from rag.plugins.summary_model import (
    SummaryGenerationTimeoutError,
    SummaryModelManager,
    SummaryModelUnavailableError,
)
from rag.storage import model_manager


class FakeCompletion:
    text = "Section summary from vLLM."


class FakeRequestOutput:
    outputs = [FakeCompletion()]


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params):
        assert prompts
        assert sampling_params.max_tokens == 384
        return [FakeRequestOutput()]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeAPIResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeRequests:
    calls = []

    @classmethod
    def post(cls, url, headers=None, json=None, timeout=None):
        cls.calls.append({"url": url, "headers": headers or {}, "json": json or {}, "timeout": timeout})
        return FakeAPIResponse({"choices": [{"message": {"content": "Section summary from API."}}]})


class HangingRequests:
    @staticmethod
    def post(url, headers=None, json=None, timeout=None):
        time.sleep(1)
        return FakeAPIResponse({"choices": [{"message": {"content": "late response"}}]})


def write_minimal_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")


def test_summary_vllm_backend_uses_configured_local_model(monkeypatch, tmp_path):
    model_path = tmp_path / "summary-model"
    write_minimal_model(model_path)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(model_path))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REPO", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_NAME", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REVISION", "main")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_AUTO_DOWNLOAD", False)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_OFFLINE_MODE", True)

    manager = SummaryModelManager(
        backend="vllm",
        model_name="Qwen/Qwen3-8B",
        max_input_tokens=4096,
        max_output_tokens=64,
        temperature=0,
        top_p=0.8,
    )

    text = manager.summarize_section(
        section_name="Method",
        content="The method uses message passing.",
        language="en",
        detail_level="short",
    )

    llm, _ = manager._get_vllm()
    assert llm.kwargs["model"] == str(model_path)
    assert llm.kwargs["max_model_len"] == 4096
    assert llm.kwargs["cpu_offload_gb"] == 0.0
    assert llm.kwargs["enforce_eager"] is True
    assert text == "Section summary from vLLM."


def test_summary_api_backend_calls_openai_compatible_endpoint(monkeypatch):
    fake_requests = types.ModuleType("requests")
    fake_requests.post = FakeRequests.post
    FakeRequests.calls = []
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    manager = SummaryModelManager(
        backend="api",
        model_name="Qwen/Qwen3-8B",
        api_base_url="http://127.0.0.1:8001/v1",
        api_key="secret",
        api_model="summary-model",
        api_timeout_seconds=12,
        max_input_tokens=4096,
        max_output_tokens=64,
        temperature=0,
        top_p=0.8,
    )

    text = manager.summarize_section(
        section_name="Method",
        content="The method uses message passing.",
        language="en",
        detail_level="short",
    )

    assert text == "Section summary from API."
    assert FakeRequests.calls
    call = FakeRequests.calls[0]
    assert call["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["json"]["model"] == "summary-model"
    assert call["json"]["max_tokens"] == 384
    assert call["json"]["stream"] is False
    assert call["timeout"] == (10.0, 12.0)


def test_summary_api_backend_has_hard_request_timeout(monkeypatch):
    fake_requests = types.ModuleType("requests")
    fake_requests.post = HangingRequests.post
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    manager = SummaryModelManager(
        backend="api",
        api_base_url="http://127.0.0.1:8001/v1",
        api_model="summary-model",
        api_timeout_seconds=5,
        api_max_retries=0,
        timeout_seconds=0.05,
    )

    started = time.monotonic()
    with pytest.raises(SummaryGenerationTimeoutError):
        manager.summarize_section(
            section_name="Method",
            content="The method uses message passing.",
            language="en",
            detail_level="short",
        )
    assert time.monotonic() - started < 0.5


def test_summary_api_backend_requires_base_url():
    manager = SummaryModelManager(backend="api")

    with pytest.raises(SummaryModelUnavailableError):
        manager.summarize_section(
            section_name="Method",
            content="The method uses message passing.",
            language="en",
            detail_level="short",
        )


def test_ensure_summary_model_uses_rag_summary_path(monkeypatch, tmp_path):
    model_path = tmp_path / "summary-model"
    write_minimal_model(model_path)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_PATH", str(model_path))
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REPO", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_NAME", "Qwen/Qwen3-8B")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_MODEL_REVISION", "main")
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_AUTO_DOWNLOAD", False)
    monkeypatch.setattr(model_manager.conf, "RAG_SUMMARY_OFFLINE_MODE", True)

    assert model_manager.ensure_summary_model() == str(model_path)


def test_summary_vllm_backend_checks_runtime_before_model_download(monkeypatch):
    sys.modules.pop("vllm", None)

    def fail_download():
        raise AssertionError("ensure_summary_model should not run before vLLM import succeeds")

    real_import = builtins.__import__

    def block_vllm_import(name, *args, **kwargs):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'vllm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(model_manager, "ensure_summary_model", fail_download)
    monkeypatch.setattr(builtins, "__import__", block_vllm_import)

    manager = SummaryModelManager(backend="vllm")

    with pytest.raises(SummaryModelUnavailableError):
        manager.summarize_section(
            section_name="Method",
            content="The method uses message passing.",
            language="en",
            detail_level="short",
        )
