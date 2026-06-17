"""Hypothetical Document Embeddings (HyDE) query expander.

Given a user query, asks an LLM to draft a short hypothetical academic
passage that would plausibly appear in a paper answering the query. The
passage is then concatenated with the original query before embedding,
so dense retrieval can match on *evidence-like* text rather than only the
short, under-specified query.

Falls back gracefully to an empty expansion on any LLM failure so that
retrieval always remains functional.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional
from urllib import error as url_error
from urllib import request as url_request

from config import conf
from rag.core.logging import get_logger

logger = get_logger(__name__)


_HYDE_SYSTEM_PROMPT = (
    "You are a scholarly retrieval assistant. Given a research question or "
    "keyword phrase, write a single compact passage (3-6 sentences, 120-220 "
    "characters in English or equivalent in Chinese) that could plausibly "
    "appear verbatim inside a peer-reviewed paper directly addressing the "
    "question. Use an academic tone, concrete technical terms, named methods "
    "and typical notation. Do NOT add disclaimers, headings, bullets or "
    "markdown. Do NOT restate that it is hypothetical. Match the language of "
    "the query (Chinese stays Chinese; English stays English)."
)


class HyDEExpander:
    """Expands a query into a hypothetical passage for dense retrieval.

    Uses the OpenAI-compatible HTTP chat endpoint configured via
    ``conf.LLM_BASE_URL`` / ``conf.LLM_MODEL_NAME``.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        temperature: float = 0.3,
        timeout_sec: int = 30,
        backend: str = "api",
        vllm_model_path: str = "",
        vllm_dtype: str = "auto",
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.85,
        vllm_max_model_len: int = 2048,
    ):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout_sec = int(timeout_sec)
        self.backend = str(backend or "api").strip().lower()
        self.vllm_model_path = str(vllm_model_path or "").strip()
        self.vllm_dtype = str(vllm_dtype or "auto").strip()
        self.vllm_tensor_parallel_size = int(vllm_tensor_parallel_size)
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        self.vllm_max_model_len = int(vllm_max_model_len)
        self._vllm = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> str:
        """Return a hypothetical passage for ``query`` or ``""`` on failure."""
        q = (query or "").strip()
        if not q:
            return ""

        messages = [
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]

        try:
            if self.backend in {"vllm", "local"}:
                passage = self._generate_vllm(q)
            else:
                passage = self._generate_api(messages)
        except Exception as exc:  # noqa: BLE001 — graceful degradation is intentional
            logger.info(f"[hyde] expansion failed, falling back to raw query: {exc}")
            return ""

        passage = (passage or "").strip()
        if not passage:
            return ""

        logger.info(f"[hyde] expanded +{len(passage)} chars")
        return passage

    # ------------------------------------------------------------------
    # Provider: OpenAI-compatible HTTP (api)
    # ------------------------------------------------------------------

    def _generate_api(self, messages: List[Dict[str, Any]]) -> str:
        """Call the LLM API with retries on transient connection failures."""
        payload = {
            "model": conf.LLM_MODEL_NAME,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = self._api_endpoint()
        headers = self._api_headers()

        last_error = None
        for attempt in range(3):
            try:
                req = url_request.Request(url=url, data=body, headers=headers, method="POST")
                with url_request.urlopen(req, timeout=self.timeout_sec) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    raw = response.read().decode(charset, errors="replace")
            except url_error.HTTPError as exc:
                # 4xx/5xx are NOT retried — auth and server-side errors are not transient.
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {detail[:200]}") from exc
            except url_error.URLError as exc:
                last_error = exc
                delay = 2 ** attempt  # 1s, 2s, 4s
                logger.info(f"[hyde] API connection attempt {attempt + 1}/3 failed (retry in {delay}s): {exc.reason}")
                time.sleep(delay)
                continue

            data = json.loads(raw)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("API response has no choices")
            message = choices[0].get("message") or {}
            return str(message.get("content") or "")

        raise RuntimeError(f"connection failed after 3 attempts: {last_error and last_error.reason}")

    @staticmethod
    def _api_endpoint() -> str:
        base_url = str(conf.LLM_BASE_URL).rstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    @staticmethod
    def _api_headers() -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = str(conf.LLM_API_KEY or "").strip()
        if api_key and api_key.lower() != "ollama":
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # ------------------------------------------------------------------
    # Provider: local vLLM
    # ------------------------------------------------------------------

    def _generate_vllm(self, query: str) -> str:
        """Generate a HyDE passage with a local vLLM engine."""
        llm, sampling_params = self._get_vllm()
        prompt = self._build_vllm_prompt(query)
        outputs = llm.generate([prompt], sampling_params)
        if not outputs:
            return ""
        generated = outputs[0].outputs[0].text if outputs[0].outputs else ""
        return str(generated or "").strip()

    @staticmethod
    def _build_vllm_prompt(query: str) -> str:
        user_prompt = (
            "Given the following academic retrieval query, write one compact "
            "paper-like passage that directly addresses it. Do not add headings, "
            "bullets, citations, or explanations.\n\n"
            f"Query: {query}\n\nPassage:"
        )
        return f"{_HYDE_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _get_vllm(self):
        if self._vllm is not None:
            return self._vllm
        if not self.vllm_model_path:
            raise ValueError("HYDE_VLLM_MODEL_PATH is required when HYDE_BACKEND=vllm")

        from vllm import LLM, SamplingParams

        logger.info("[hyde] loading vLLM model path=%s", self.vllm_model_path)
        llm = LLM(
            model=self.vllm_model_path,
            dtype=self.vllm_dtype,
            trust_remote_code=True,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            max_model_len=self.vllm_max_model_len,
        )
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=max(self.temperature, 0.0),
            top_p=0.9,
            stop=["\n\nQuery:", "\n\n###", "\nQuery:"],
        )
        self._vllm = (llm, sampling_params)
        return self._vllm

# Module-level singleton: HyDE is stateless per-call, so one instance is enough.
_singleton: Optional[HyDEExpander] = None
_singleton_lock = threading.Lock()


def get_hyde_expander() -> HyDEExpander:
    """Return a lazily-constructed process-wide :class:`HyDEExpander`."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = HyDEExpander(
                max_tokens=int(getattr(conf, "HYDE_MAX_TOKENS", 256)),
                temperature=float(getattr(conf, "HYDE_TEMPERATURE", 0.3)),
                timeout_sec=int(getattr(conf, "HYDE_API_TIMEOUT", 60)),
                backend=str(getattr(conf, "HYDE_BACKEND", "api")),
                vllm_model_path=str(getattr(conf, "HYDE_VLLM_MODEL_PATH", "")),
                vllm_dtype=str(getattr(conf, "HYDE_VLLM_DTYPE", "auto")),
                vllm_tensor_parallel_size=int(getattr(conf, "HYDE_VLLM_TENSOR_PARALLEL_SIZE", 1)),
                vllm_gpu_memory_utilization=float(getattr(conf, "HYDE_VLLM_GPU_MEMORY_UTILIZATION", 0.85)),
                vllm_max_model_len=int(getattr(conf, "HYDE_VLLM_MAX_MODEL_LEN", 2048)),
            )
        return _singleton
