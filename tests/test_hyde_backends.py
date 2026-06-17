import sys
import types

from rag.plugins import HyDEExpander


class FakeCompletion:
    text = "A vLLM hypothetical passage."


class FakeRequestOutput:
    outputs = [FakeCompletion()]


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params):
        assert prompts
        assert sampling_params.max_tokens == 8
        return [FakeRequestOutput()]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_hyde_vllm_backend_uses_vllm(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    expander = HyDEExpander(
        backend="vllm",
        vllm_model_path="local-model",
        max_tokens=8,
        temperature=0,
    )

    assert expander.expand("SCMA detection") == "A vLLM hypothetical passage."


def test_hyde_api_backend_can_be_mocked():
    expander = HyDEExpander(backend="api")
    expander._generate_api = lambda messages: "API hypothetical passage."

    assert expander.expand("query") == "API hypothetical passage."
