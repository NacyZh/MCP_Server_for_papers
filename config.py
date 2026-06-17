"""Validated configuration for the ScholarAgent RAG MCP server."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.core.logging import get_logger

if __package__ is None:
    _proj = Path(__file__).resolve().parent
    while _proj and not (_proj / "pyproject.toml").exists():
        _proj = _proj.parent
    if _proj and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

logger = get_logger(__name__)


def _resolve_base_relative(path_value: str | os.PathLike[str]) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


class Config(BaseSettings):
    """Global RAG MCP configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    PROJECT_ROOT: str = Field(default_factory=lambda: str(PROJECT_ROOT))

    WORKSPACE_DIR: str = "workspace"
    DB_DIR: str = "./rag/db"
    MODELS_DIR: str = "./rag/models"
    PAPERS_DIR: str = ""

    BGE_M3_MODEL_PATH: str = ""
    BGE_RERANKER_MODEL_PATH: str = ""
    BGE_AUTO_DOWNLOAD: bool = True
    BGE_OFFLINE_MODE: bool = False
    BGE_M3_MODEL_REPO: str = "BAAI/bge-m3"
    BGE_RERANKER_MODEL_REPO: str = "BAAI/bge-reranker-v2-m3"
    BGE_M3_MODEL_REVISION: str = "main"
    BGE_RERANKER_MODEL_REVISION: str = "main"
    BGE_MODEL_LOCK_TIMEOUT_SEC: float = Field(default=600.0, gt=0.0)
    BGE_MODEL_LOCK_STALE_SEC: float = Field(default=3600.0, gt=0.0)
    PAPER_PARSER_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"

    ENABLE_HYDE: bool = True
    HYDE_BACKEND: Literal["api", "vllm"] = "api"
    HYDE_VLLM_MODEL_PATH: str = "./rag/finetune/models/hyde-qwen-lora"
    HYDE_VLLM_DTYPE: str = "auto"
    HYDE_VLLM_TENSOR_PARALLEL_SIZE: int = Field(default=1, ge=1)
    HYDE_VLLM_GPU_MEMORY_UTILIZATION: float = Field(default=0.85, gt=0.0, le=1.0)
    HYDE_VLLM_MAX_MODEL_LEN: int = Field(default=2048, ge=1)
    HYDE_MAX_TOKENS: int = Field(default=256, ge=1)
    HYDE_TEMPERATURE: float = Field(default=0.1, ge=0.0)

    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL_NAME: str = ""
    HYDE_API_TIMEOUT: float = Field(default=60.0, gt=0.0)

    RAG_SUMMARY_MODEL_NAME: str = "Qwen/Qwen3-8B-AWQ"
    RAG_SUMMARY_BACKEND: Literal["api", "vllm"] = "api"
    RAG_SUMMARY_API_BASE_URL: str = ""
    RAG_SUMMARY_API_KEY: str = ""
    RAG_SUMMARY_API_MODEL: str = ""
    RAG_SUMMARY_API_TIMEOUT_SECONDS: float = Field(default=180.0, gt=0.0)
    RAG_SUMMARY_API_MAX_RETRIES: int = Field(default=2, ge=0, le=10)
    RAG_SUMMARY_API_HEALTH_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0.0)
    RAG_SUMMARY_MODEL_PATH: str = "./rag/models/Qwen3-8B-AWQ"
    RAG_SUMMARY_MODEL_REPO: str = "Qwen/Qwen3-8B-AWQ"
    RAG_SUMMARY_MODEL_REVISION: str = "main"
    RAG_SUMMARY_AUTO_DOWNLOAD: bool = True
    RAG_SUMMARY_OFFLINE_MODE: bool = False
    RAG_SUMMARY_DEVICE: Literal["auto", "cpu", "cuda"] = "cuda"
    RAG_SUMMARY_DTYPE: str = "auto"
    RAG_SUMMARY_TENSOR_PARALLEL_SIZE: int = Field(default=1, ge=1)
    RAG_SUMMARY_GPU_MEMORY_UTILIZATION: float = Field(default=0.75, gt=0.0, le=1.0)
    RAG_SUMMARY_CPU_OFFLOAD_GB: float = Field(default=0.0, ge=0.0)
    RAG_SUMMARY_MAX_INPUT_TOKENS: int = Field(default=4096, ge=1024)
    RAG_SUMMARY_MAX_OUTPUT_TOKENS: int = Field(default=2048, ge=128)
    RAG_SUMMARY_TEMPERATURE: float = Field(default=0.2, ge=0.0)
    RAG_SUMMARY_TOP_P: float = Field(default=0.8, gt=0.0, le=1.0)
    RAG_SUMMARY_CONCURRENCY: int = Field(default=1, ge=1)
    RAG_SUMMARY_TIMEOUT_SECONDS: float = Field(default=300.0, gt=0.0)
    RAG_SUMMARY_PROMPT_VERSION: str = "qwen3-summary-v1"
    RAG_SUMMARY_ENFORCE_EAGER: bool = True

    TOOL_MAX_QUERY_CHARS: int = Field(default=800, ge=1, le=10000)
    TOOL_MAX_SECTION_CHARS: int = Field(default=120, ge=1, le=1000)
    TOOL_MAX_RETURN_CHARS: int = Field(default=20000, ge=1000, le=200000)
    TOOL_MAX_PAPER_IDS: int = Field(default=20, ge=1, le=100)
    TOOL_JOB_STATUS_WAIT_INITIAL_SECONDS: float = Field(default=10.0, ge=0.0, le=60.0)
    TOOL_JOB_STATUS_WAIT_STEP_SECONDS: float = Field(default=10.0, ge=0.0, le=60.0)
    TOOL_JOB_STATUS_WAIT_MAX_SECONDS: float = Field(default=120.0, ge=0.0, le=120.0)
    RETRIEVAL_EVAL_DATASET_PATH: str = "./rag/evaluation/data/retrieval_eval.jsonl"
    RETRIEVAL_EVAL_RESULTS_DIR: str = "./rag/evaluation/results"
    RETRIEVAL_EVAL_MAX_CASES: int = Field(default=200, ge=1, le=10000)

    LOG_DIR: str = Field(default="", validation_alias="RAG_LOG_DIR")
    LOG_FILE: str = Field(default="", validation_alias="RAG_LOG_FILE")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias="RAG_LOG_LEVEL",
    )
    LOG_TO_CONSOLE: bool = Field(default=True, validation_alias="RAG_LOG_TO_CONSOLE")
    LOG_MAX_BYTES: int = Field(default=10 * 1024 * 1024, ge=1, validation_alias="RAG_LOG_MAX_BYTES")
    LOG_BACKUP_COUNT: int = Field(default=5, ge=0, validation_alias="RAG_LOG_BACKUP_COUNT")

    DEFAULT_PAPER_TITLE: str = "Unknown"

    @field_validator(
        "WORKSPACE_DIR",
        "DB_DIR",
        "MODELS_DIR",
        "PAPERS_DIR",
        "BGE_M3_MODEL_PATH",
        "BGE_RERANKER_MODEL_PATH",
        "HYDE_VLLM_MODEL_PATH",
        "RAG_SUMMARY_MODEL_PATH",
        "LOG_DIR",
        "LOG_FILE",
        "RETRIEVAL_EVAL_DATASET_PATH",
        "RETRIEVAL_EVAL_RESULTS_DIR",
        mode="before",
    )
    @classmethod
    def _strip_path_value(cls, value):
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _normalize_log_level(cls, value):
        return str(value or "INFO").strip().upper()

    @field_validator(
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "RAG_SUMMARY_API_BASE_URL",
        "RAG_SUMMARY_API_KEY",
        "RAG_SUMMARY_API_MODEL",
        mode="before",
    )
    @classmethod
    def _strip_api_value(cls, value):
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _derive_and_resolve_paths(self):
        self.WORKSPACE_DIR = _resolve_base_relative(self.WORKSPACE_DIR)
        self.DB_DIR = _resolve_base_relative(self.DB_DIR)
        self.MODELS_DIR = _resolve_base_relative(self.MODELS_DIR)
        self.PAPERS_DIR = _resolve_base_relative(self.PAPERS_DIR or os.path.join(self.WORKSPACE_DIR, "papers"))

        self.BGE_M3_MODEL_PATH = _resolve_base_relative(
            self.BGE_M3_MODEL_PATH or os.path.join(self.MODELS_DIR, "bge-m3")
        )
        self.BGE_RERANKER_MODEL_PATH = _resolve_base_relative(
            self.BGE_RERANKER_MODEL_PATH or os.path.join(self.MODELS_DIR, "bge-reranker-v2-m3")
        )
        self.HYDE_VLLM_MODEL_PATH = _resolve_base_relative(self.HYDE_VLLM_MODEL_PATH)
        self.RAG_SUMMARY_MODEL_PATH = _resolve_base_relative(self.RAG_SUMMARY_MODEL_PATH)
        self.LOG_DIR = _resolve_base_relative(self.LOG_DIR or os.path.join("rag", "logs"))
        self.LOG_FILE = _resolve_base_relative(self.LOG_FILE or os.path.join(self.LOG_DIR, "rag.log"))
        self.RETRIEVAL_EVAL_DATASET_PATH = _resolve_base_relative(self.RETRIEVAL_EVAL_DATASET_PATH)
        self.RETRIEVAL_EVAL_RESULTS_DIR = _resolve_base_relative(self.RETRIEVAL_EVAL_RESULTS_DIR)

        self.RAG_SUMMARY_API_BASE_URL = self.RAG_SUMMARY_API_BASE_URL or self.LLM_BASE_URL
        self.RAG_SUMMARY_API_KEY = self.RAG_SUMMARY_API_KEY or self.LLM_API_KEY
        self.RAG_SUMMARY_API_MODEL = self.RAG_SUMMARY_API_MODEL or self.LLM_MODEL_NAME or self.RAG_SUMMARY_MODEL_NAME

        if self.ENABLE_HYDE and self.HYDE_BACKEND == "vllm" and not self.HYDE_VLLM_MODEL_PATH:
            raise ValueError("HYDE_VLLM_MODEL_PATH is required when HYDE_BACKEND=vllm")
        if self.RAG_SUMMARY_BACKEND == "vllm" and not self.RAG_SUMMARY_MODEL_PATH:
            raise ValueError("RAG_SUMMARY_MODEL_PATH is required when summary generation is enabled")
        return self

    def resolve_api_key(self, api_key: str | None = None, base_url: str | None = None) -> str:
        value = api_key if api_key is not None else self.LLM_API_KEY
        if (value or "").lower() == "ollama":
            return "not-needed"
        if "ollama" in (base_url or self.LLM_BASE_URL or "").lower():
            return "not-needed"
        return value or ""

    def check_config(self) -> None:
        """Create required runtime directories."""
        for path in (self.DB_DIR, self.PAPERS_DIR, self.MODELS_DIR, self.LOG_DIR, self.RETRIEVAL_EVAL_RESULTS_DIR):
            os.makedirs(path, exist_ok=True)
        logger.info("[config] project_root=%s", self.PROJECT_ROOT)


conf = Config()


if __name__ == "__main__":
    conf.check_config()
