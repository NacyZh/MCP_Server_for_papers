"""Local summary generation backend for cached paper summaries."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from typing import Any, Dict, List, Optional

from config import conf
from rag.core.logging import get_logger
from rag.storage import model_manager

logger = get_logger(__name__)


class SummaryModelError(RuntimeError):
    """Base class for summary model failures."""


class SummaryModelUnavailableError(SummaryModelError):
    """Raised when the configured summary model cannot be loaded."""


class SummaryGenerationTimeoutError(SummaryModelError, TimeoutError):
    """Raised when generation exceeds the configured timeout."""


class SummaryGenerationError(SummaryModelError):
    """Raised when the model output cannot be used."""


class SummaryModelManager:
    """Lazy local generator for paper profile and summary cache jobs."""

    def __init__(
        self,
        *,
        backend: str = "api",
        model_name: str = "Qwen/Qwen3-8B",
        api_base_url: str = "",
        api_key: str = "",
        api_model: str = "",
        api_timeout_seconds: float = 180.0,
        api_max_retries: int = 2,
        prompt_version: str = "qwen3-summary-v1",
        max_input_tokens: int = 4096,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.8,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        cpu_offload_gb: float = 0.0,
        enforce_eager: bool = True,
        concurrency: int = 1,
        timeout_seconds: float = 300.0,
    ):
        self.backend = str(backend or "api").strip().lower()
        self.model_name = str(model_name or "Qwen/Qwen3-8B").strip()
        self.api_base_url = str(api_base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.api_model = str(api_model or self.model_name).strip()
        self.api_timeout_seconds = float(api_timeout_seconds)
        self.api_max_retries = int(api_max_retries)
        self.prompt_version = str(prompt_version or "qwen3-summary-v1").strip()
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.dtype = str(dtype or "auto").strip()
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.cpu_offload_gb = float(cpu_offload_gb)
        self.enforce_eager = bool(enforce_eager)
        self.timeout_seconds = float(timeout_seconds)
        self._llm = None
        self._llm_lock = threading.Lock()
        self._generation_sem = threading.BoundedSemaphore(max(1, int(concurrency)))

    def summarize_section(
        self,
        *,
        section_name: str,
        content: str,
        language: str,
        detail_level: str,
    ) -> str:
        prompt = self._section_prompt(section_name, content, language, detail_level)
        text = self._generate(prompt, max_tokens=self._section_output_tokens(detail_level))
        return self._strip_reasoning(text)

    def build_profile(
        self,
        *,
        paper: Dict[str, Any],
        section_summaries: List[Dict[str, str]],
        language: str,
    ) -> Dict[str, Any]:
        prompt = self._profile_prompt(paper, section_summaries, language)
        text = self._generate(prompt, max_tokens=min(self.max_output_tokens, 1200))
        data = _parse_json_object(text)
        if not data:
            raise SummaryGenerationError("summary profile output was not valid JSON")
        data.setdefault("paper_id", paper.get("id") or "")
        data.setdefault("title", paper.get("title") or "")
        data.setdefault("language", language)
        for key in ("problem", "background", "method", "experiments", "limitations"):
            data[key] = str(data.get(key) or "")
        contributions = data.get("contributions")
        if not isinstance(contributions, list):
            data["contributions"] = [str(contributions)] if contributions else []
        keywords = data.get("keywords")
        if not isinstance(keywords, list):
            data["keywords"] = [str(keywords)] if keywords else []
        return data

    def build_paper_summary(
        self,
        *,
        profile: Dict[str, Any],
        section_summaries: List[Dict[str, str]],
        language: str,
        detail_level: str,
    ) -> Dict[str, str]:
        prompt = self._summary_prompt(profile, section_summaries, language, detail_level)
        text = self._generate(prompt, max_tokens=self._summary_output_tokens(detail_level))
        data = _parse_json_object(text)
        if not data:
            cleaned = self._strip_reasoning(text)
            data = {"results": cleaned}
        result = {}
        for key in ("background", "problem", "method", "experiments", "results", "limitations", "takeaways"):
            result[key] = str(data.get(key) or "")
        return result

    def _generate(self, prompt: str, *, max_tokens: int) -> str:
        if self.backend == "api":
            return self._generate_api(prompt, max_tokens=max_tokens)
        if self.backend == "vllm":
            return self._generate_vllm(prompt, max_tokens=max_tokens)
        raise SummaryModelUnavailableError(f"Unsupported summary backend: {self.backend}")

    def _generate_api(self, prompt: str, *, max_tokens: int) -> str:
        if not self.api_base_url:
            raise SummaryModelUnavailableError("RAG_SUMMARY_API_BASE_URL is required when RAG_SUMMARY_BACKEND=api")
        endpoint = _chat_completions_endpoint(self.api_base_url)
        payload = {
            "model": self.api_model or self.model_name,
            "messages": [{"role": "user", "content": self._truncate_prompt(prompt)}],
            "max_tokens": max(1, int(max_tokens)),
            "temperature": max(self.temperature, 0.0),
            "top_p": self.top_p,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.monotonic()
        deadline = started + max(0.01, self.timeout_seconds)
        last_error: Exception | None = None
        with self._generation_sem:
            for attempt in range(max(0, self.api_max_retries) + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SummaryGenerationTimeoutError(
                        f"Summary generation exceeded timeout: {self.timeout_seconds:.1f}s"
                    )
                call_timeout = max(0.1, min(self.api_timeout_seconds, remaining))
                try:
                    response = _post_json_with_hard_timeout(
                        endpoint,
                        headers=headers,
                        payload=payload,
                        timeout_seconds=call_timeout,
                    )
                    if response.status_code >= 500 and attempt < self.api_max_retries:
                        last_error = SummaryGenerationError(
                            f"summary API returned HTTP {response.status_code}: {response.text[:500]}"
                        )
                        time.sleep(min(2.0, 0.25 * (attempt + 1)))
                        continue
                    response.raise_for_status()
                    data = response.json()
                    text = _extract_chat_completion_text(data)
                    break
                except SummaryGenerationTimeoutError as exc:
                    last_error = exc
                    if attempt >= self.api_max_retries or time.monotonic() >= deadline:
                        raise
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.api_max_retries:
                        raise SummaryGenerationError(f"Summary API generation failed: {exc}") from exc
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
            else:
                raise SummaryGenerationError(f"Summary API generation failed: {last_error}")

        elapsed = time.monotonic() - started
        if elapsed > self.timeout_seconds:
            raise SummaryGenerationTimeoutError(
                f"Summary generation exceeded timeout: {elapsed:.1f}s > {self.timeout_seconds:.1f}s"
            )
        text = str(text or "").strip()
        if not text:
            raise SummaryGenerationError("summary API returned no output")
        return text

    def _generate_vllm(self, prompt: str, *, max_tokens: int) -> str:
        started = time.monotonic()
        with self._generation_sem:
            llm, sampling_params_cls = self._get_vllm()
            sampling_params = sampling_params_cls(
                max_tokens=max(1, int(max_tokens)),
                temperature=max(self.temperature, 0.0),
                top_p=self.top_p,
                stop=["<|im_end|>", "\n\nUser:", "\n\nSystem:"],
            )
            outputs = llm.generate([self._truncate_prompt(prompt)], sampling_params)
        elapsed = time.monotonic() - started
        if elapsed > self.timeout_seconds:
            raise SummaryGenerationTimeoutError(
                f"Summary generation exceeded timeout: {elapsed:.1f}s > {self.timeout_seconds:.1f}s"
            )
        if not outputs or not outputs[0].outputs:
            raise SummaryGenerationError("summary model returned no output")
        return str(outputs[0].outputs[0].text or "").strip()

    def _get_vllm(self):
        if self.backend != "vllm":
            raise SummaryModelUnavailableError(f"Unsupported summary backend: {self.backend}")
        if self._llm is not None:
            return self._llm
        with self._llm_lock:
            if self._llm is not None:
                return self._llm
            try:
                from vllm import LLM, SamplingParams

                model_path = model_manager.ensure_summary_model()
                logger.info(
                    "[summary] loading vLLM model path=%s model_name=%s max_model_len=%s",
                    model_path,
                    self.model_name,
                    self.max_input_tokens,
                )
                llm = LLM(
                    model=model_path,
                    dtype=self.dtype,
                    trust_remote_code=True,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    cpu_offload_gb=self.cpu_offload_gb,
                    max_model_len=self.max_input_tokens,
                    enforce_eager=self.enforce_eager,
                )
            except Exception as exc:
                raise SummaryModelUnavailableError(f"Summary vLLM model is unavailable: {exc}") from exc
            self._llm = (llm, SamplingParams)
            return self._llm

    def _truncate_prompt(self, prompt: str) -> str:
        # Approximate token control without importing tokenizer at request time.
        char_limit = max(4000, self.max_input_tokens * 4)
        value = str(prompt or "")
        if len(value) <= char_limit:
            return value
        head = value[: char_limit // 4]
        tail = value[-(char_limit - len(head)) :]
        return f"{head}\n\n[...content truncated for summary model input budget...]\n\n{tail}"

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        value = str(text or "").strip()
        value = re.sub(r"(?is)<think>.*?</think>", "", value).strip()
        value = _strip_code_fence(value)
        reasoning_cues = (
            "\nOkay,",
            "\nOkay, I",
            "\nLet me",
            "\nI need to",
            "\nWe need to",
            "\nThe task is",
        )
        cut_points = [value.find(cue) for cue in reasoning_cues if value.find(cue) > 0]
        if cut_points:
            value = value[: min(cut_points)].strip()
        return value

    @staticmethod
    def _section_output_tokens(detail_level: str) -> int:
        return {"short": 384, "medium": 768, "long": 1200}.get(detail_level, 768)

    def _summary_output_tokens(self, detail_level: str) -> int:
        return min(self.max_output_tokens, {"short": 700, "medium": 1400, "long": 2048}.get(detail_level, 1400))

    @staticmethod
    def _language_instruction(language: str) -> str:
        if language == "zh":
            return "Write in concise academic Chinese."
        return "Write in concise academic English."

    def _section_prompt(self, section_name: str, content: str, language: str, detail_level: str) -> str:
        return (
            "System: You summarize academic paper sections for a local RAG cache. "
            "Output only the final summary. Do not include hidden reasoning, planning, self-reflection, "
            "markdown tables, citations you cannot verify, or invented facts. /no_think\n\n"
            f"Task: Summarize the section named {section_name!r}. "
            f"{self._language_instruction(language)} Detail level: {detail_level}. "
            "Preserve concrete methods, equations or algorithm names, experimental setup, datasets, metrics, "
            "and limitations when present. If information is absent, omit it.\n\n"
            f"Section text:\n{content}\n\nSummary:"
        )

    def _profile_prompt(
        self,
        paper: Dict[str, Any],
        section_summaries: List[Dict[str, str]],
        language: str,
    ) -> str:
        sections = _format_section_summaries(section_summaries)
        return (
            "System: Build a compact structured profile of an academic paper from cached section summaries. "
            "Return strict JSON only. Do not include markdown, planning, self-reflection, or reasoning. /no_think\n\n"
            f"Language: {language}. {self._language_instruction(language)}\n"
            f"Paper ID: {paper.get('id')}\nTitle: {paper.get('title')}\nAuthors: {paper.get('authors') or ''}\n\n"
            "Required JSON schema:\n"
            '{"problem":"","background":"","method":"","contributions":[""],'
            '"experiments":"","limitations":"","keywords":[""]}\n\n'
            f"Section summaries:\n{sections}\n\nJSON:"
        )

    def _summary_prompt(
        self,
        profile: Dict[str, Any],
        section_summaries: List[Dict[str, str]],
        language: str,
        detail_level: str,
    ) -> str:
        sections = _format_section_summaries(section_summaries)
        return (
            "System: Build a structured paper summary from a cached paper profile and section summaries. "
            "Return strict JSON only. Do not include markdown, planning, self-reflection, or reasoning. "
            "Do not invent evidence. /no_think\n\n"
            f"Language: {language}. {self._language_instruction(language)} Detail level: {detail_level}.\n"
            "Required JSON schema:\n"
            '{"background":"","problem":"","method":"","experiments":"","results":"","limitations":"","takeaways":""}'
            "\n\n"
            f"Profile JSON:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
            f"Section summaries:\n{sections}\n\nJSON:"
        )


def _format_section_summaries(section_summaries: List[Dict[str, str]]) -> str:
    lines = []
    for item in section_summaries:
        name = item.get("section_name") or item.get("section") or "Unknown"
        summary = item.get("summary") or item.get("summary_text") or ""
        lines.append(f"[{name}]\n{summary}")
    return "\n\n".join(lines)


def _chat_completions_endpoint(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def _post_json_with_hard_timeout(
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: float,
):
    """Run requests.post behind a hard wall-clock timeout.

    requests' timeout is socket inactivity based and is not a strict total
    deadline. A stuck API call must not keep an MCP background job in running
    state forever, so the job thread waits on a daemon worker for a bounded
    amount of time and raises even if the socket layer is still blocked.
    """
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def run_request() -> None:
        try:
            import requests

            connect_timeout = min(10.0, max(0.1, float(timeout_seconds)))
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout, max(0.1, float(timeout_seconds))),
            )
            result_queue.put(("response", response))
        except Exception as exc:
            result_queue.put(("error", exc))

    worker = threading.Thread(target=run_request, name="summary-api-request", daemon=True)
    worker.start()
    try:
        kind, value = result_queue.get(timeout=max(0.1, float(timeout_seconds)))
    except queue.Empty as exc:
        raise SummaryGenerationTimeoutError(
            f"Summary API request exceeded timeout: {timeout_seconds:.1f}s"
        ) from exc
    if kind == "error":
        raise value
    return value


def _extract_chat_completion_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _strip_code_fence(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value).strip()
    return value


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = _strip_code_fence(re.sub(r"(?is)<think>.*?</think>", "", str(text or ""))).strip()
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


_singleton: Optional[SummaryModelManager] = None
_singleton_lock = threading.Lock()


def get_summary_model_manager() -> SummaryModelManager:
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SummaryModelManager(
                backend=conf.RAG_SUMMARY_BACKEND,
                model_name=conf.RAG_SUMMARY_MODEL_NAME,
                api_base_url=conf.RAG_SUMMARY_API_BASE_URL,
                api_key=conf.resolve_api_key(
                    api_key=conf.RAG_SUMMARY_API_KEY,
                    base_url=conf.RAG_SUMMARY_API_BASE_URL,
                ),
                api_model=conf.RAG_SUMMARY_API_MODEL,
                api_timeout_seconds=conf.RAG_SUMMARY_API_TIMEOUT_SECONDS,
                api_max_retries=conf.RAG_SUMMARY_API_MAX_RETRIES,
                prompt_version=conf.RAG_SUMMARY_PROMPT_VERSION,
                max_input_tokens=conf.RAG_SUMMARY_MAX_INPUT_TOKENS,
                max_output_tokens=conf.RAG_SUMMARY_MAX_OUTPUT_TOKENS,
                temperature=conf.RAG_SUMMARY_TEMPERATURE,
                top_p=conf.RAG_SUMMARY_TOP_P,
                dtype=conf.RAG_SUMMARY_DTYPE,
                tensor_parallel_size=conf.RAG_SUMMARY_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=conf.RAG_SUMMARY_GPU_MEMORY_UTILIZATION,
                cpu_offload_gb=conf.RAG_SUMMARY_CPU_OFFLOAD_GB,
                enforce_eager=conf.RAG_SUMMARY_ENFORCE_EAGER,
                concurrency=conf.RAG_SUMMARY_CONCURRENCY,
                timeout_seconds=conf.RAG_SUMMARY_TIMEOUT_SECONDS,
            )
        return _singleton
