"""Central configuration loaded from environment variables and .env file."""

import os
import sys
from pathlib import Path

if __package__ is None:
    _proj = Path(__file__).resolve().parent
    while _proj and not (_proj / "pyproject.toml").exists():
        _proj = _proj.parent
    if _proj and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

from dotenv import load_dotenv

load_dotenv()

from scholar_agent.core.logging import get_logger

logger = get_logger(__name__)


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_base_relative(path_value: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(project_root, path_value))


class Config:
    """Global configuration singleton loaded from environment variables."""

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    WORKSPACE_DIR = _resolve_base_relative(os.getenv("WORKSPACE_DIR", "workspace"))
    DB_DIR = os.path.join(WORKSPACE_DIR, "db")
    MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
    PAPERS_DIR = _resolve_base_relative(
        os.getenv("PAPERS_DIR", os.path.join(WORKSPACE_DIR, "papers"))
    )
    BGE_M3_MODEL_PATH = _resolve_base_relative(
        os.getenv("BGE_M3_MODEL_PATH", os.path.join(MODELS_DIR, "bge-m3"))
    )
    BGE_RERANKER_MODEL_PATH = _resolve_base_relative(
        os.getenv("BGE_RERANKER_MODEL_PATH", os.path.join(MODELS_DIR, "bge-reranker-v2-m3"))
    )
    PAPER_PARSER_DEVICE = os.getenv("PAPER_PARSER_DEVICE", "auto")

    # ---- LLM provider ----
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

    # ---- Multi-Agent: per-agent LLM configuration ----
    # Each agent can use a different model / backend. Defaults to the main LLM
    # config set above when the per-agent override is empty.
    #
    # Supervisor (routing + synthesis)
    AGENT_SUPERVISOR_MODEL = os.getenv("AGENT_SUPERVISOR_MODEL", "") or LLM_MODEL_NAME
    AGENT_SUPERVISOR_BASE_URL = os.getenv("AGENT_SUPERVISOR_BASE_URL", "") or LLM_BASE_URL
    AGENT_SUPERVISOR_API_KEY = os.getenv("AGENT_SUPERVISOR_API_KEY", "") or LLM_API_KEY
    # Summarizer (long-context structured reading)
    AGENT_SUMMARIZER_MODEL = os.getenv("AGENT_SUMMARIZER_MODEL", "") or LLM_MODEL_NAME
    AGENT_SUMMARIZER_BASE_URL = os.getenv("AGENT_SUMMARIZER_BASE_URL", "") or LLM_BASE_URL
    AGENT_SUMMARIZER_API_KEY = os.getenv("AGENT_SUMMARIZER_API_KEY", "") or LLM_API_KEY
    # Methodology Analyst (math / algorithm extraction)
    AGENT_METHODOLOGY_MODEL = os.getenv("AGENT_METHODOLOGY_MODEL", "") or LLM_MODEL_NAME
    AGENT_METHODOLOGY_BASE_URL = os.getenv("AGENT_METHODOLOGY_BASE_URL", "") or LLM_BASE_URL
    AGENT_METHODOLOGY_API_KEY = os.getenv("AGENT_METHODOLOGY_API_KEY", "") or LLM_API_KEY
    # Code Builder (code generation)
    AGENT_CODE_BUILDER_MODEL = os.getenv("AGENT_CODE_BUILDER_MODEL", "") or LLM_MODEL_NAME
    AGENT_CODE_BUILDER_BASE_URL = os.getenv("AGENT_CODE_BUILDER_BASE_URL", "") or LLM_BASE_URL
    AGENT_CODE_BUILDER_API_KEY = os.getenv("AGENT_CODE_BUILDER_API_KEY", "") or LLM_API_KEY
    # Writing Editor (paper drafting, rewriting, and polishing)
    AGENT_WRITING_EDITOR_MODEL = os.getenv("AGENT_WRITING_EDITOR_MODEL", "") or LLM_MODEL_NAME
    AGENT_WRITING_EDITOR_BASE_URL = os.getenv("AGENT_WRITING_EDITOR_BASE_URL", "") or LLM_BASE_URL
    AGENT_WRITING_EDITOR_API_KEY = os.getenv("AGENT_WRITING_EDITOR_API_KEY", "") or LLM_API_KEY
    # Literature Searcher (lightweight search + relevance judge)
    AGENT_LITERATURE_MODEL = os.getenv("AGENT_LITERATURE_MODEL", "") or LLM_MODEL_NAME
    AGENT_LITERATURE_BASE_URL = os.getenv("AGENT_LITERATURE_BASE_URL", "") or LLM_BASE_URL
    AGENT_LITERATURE_API_KEY = os.getenv("AGENT_LITERATURE_API_KEY", "") or LLM_API_KEY

    # ---- Retrieval: HyDE query expansion ----
    ENABLE_HYDE = _get_bool_env("ENABLE_HYDE", True)
    HYDE_MAX_TOKENS = _get_int_env("HYDE_MAX_TOKENS", 256)
    HYDE_TEMPERATURE = _get_float_env("HYDE_TEMPERATURE", 0.1)

    # ---- Agent LLM default parameters (overridable via env vars) ----
    AGENT_SUPERVISOR_TEMPERATURE = _get_float_env("AGENT_SUPERVISOR_TEMPERATURE", 0.2)
    AGENT_LLM_TIMEOUT = _get_float_env("AGENT_LLM_TIMEOUT", 60.0)
    AGENT_SUPERVISOR_MAX_TOKENS = _get_int_env("AGENT_SUPERVISOR_MAX_TOKENS", 2048)
    AGENT_SUPERVISOR_SYNTHESIS_TEMPERATURE = _get_float_env("AGENT_SUPERVISOR_SYNTHESIS_TEMPERATURE", 0.3)
    AGENT_SUPERVISOR_SYNTHESIS_MAX_TOKENS = _get_int_env("AGENT_SUPERVISOR_SYNTHESIS_MAX_TOKENS", 3072)
    AGENT_SUMMARIZER_TEMPERATURE = _get_float_env("AGENT_SUMMARIZER_TEMPERATURE", 0.3)
    AGENT_SUMMARIZER_MAX_TOKENS = _get_int_env("AGENT_SUMMARIZER_MAX_TOKENS", 4096)
    AGENT_METHODOLOGY_TEMPERATURE = _get_float_env("AGENT_METHODOLOGY_TEMPERATURE", 0.2)
    AGENT_METHODOLOGY_MAX_TOKENS = _get_int_env("AGENT_METHODOLOGY_MAX_TOKENS", 4096)
    AGENT_CODE_BUILDER_TEMPERATURE = _get_float_env("AGENT_CODE_BUILDER_TEMPERATURE", 0.1)
    AGENT_CODE_BUILDER_MAX_TOKENS = _get_int_env("AGENT_CODE_BUILDER_MAX_TOKENS", 8192)
    AGENT_WRITING_EDITOR_TEMPERATURE = _get_float_env("AGENT_WRITING_EDITOR_TEMPERATURE", 0.2)
    AGENT_WRITING_EDITOR_MAX_TOKENS = _get_int_env("AGENT_WRITING_EDITOR_MAX_TOKENS", 8192)
    AGENT_LITERATURE_TEMPERATURE = _get_float_env("AGENT_LITERATURE_TEMPERATURE", 0.3)
    AGENT_LITERATURE_MAX_TOKENS = _get_int_env("AGENT_LITERATURE_MAX_TOKENS", 3072)

    # ---- Agent routing constants ----
    VALID_AGENTS = frozenset({
        "summarizer",
        "methodology",
        "code_builder",
        "literature",
        "database_manager",
        "writing_editor",
        "FINISH",
    })
    EXPERT_NAMES = frozenset({
        "summarizer",
        "methodology",
        "code_builder",
        "literature",
        "database_manager",
        "writing_editor",
    })

    # ---- Truncation / context limits ----
    SUPERVISOR_EXPERT_OUTPUT_TRUNC = _get_int_env("SUPERVISOR_EXPERT_OUTPUT_TRUNC", 600)
    SUPERVISOR_MESSAGE_TRUNC = _get_int_env("SUPERVISOR_MESSAGE_TRUNC", 800)
    SUPERVISOR_TOOL_RESULT_TRUNC = _get_int_env("SUPERVISOR_TOOL_RESULT_TRUNC", 600)
    SUPERVISOR_SYNTHESIS_CONTENT_TRUNC = _get_int_env("SUPERVISOR_SYNTHESIS_CONTENT_TRUNC", 4000)
    EXPERT_MESSAGE_OUTPUT_TRUNC = _get_int_env("EXPERT_MESSAGE_OUTPUT_TRUNC", 3000)
    EXPERT_CONTEXT_MAX_CHARS_PER_PAPER = _get_int_env("EXPERT_CONTEXT_MAX_CHARS_PER_PAPER", 8000)
    EXPERT_METHODOLOGY_MAX_CHARS_PER_PAPER = _get_int_env("EXPERT_METHODOLOGY_MAX_CHARS_PER_PAPER", 10000)
    EXPERT_LITERATURE_MAX_LOCAL_PREVIEW = _get_int_env("EXPERT_LITERATURE_MAX_LOCAL_PREVIEW", 15)
    EXPERT_LITERATURE_ARXIV_MAX_RESULTS = _get_int_env("EXPERT_LITERATURE_ARXIV_MAX_RESULTS", 5)
    AGENT_CODE_BUILDER_AUTONOMOUS = _get_bool_env("AGENT_CODE_BUILDER_AUTONOMOUS", True)
    SCHOLAR_AGENT_WORK_ROOT = os.getenv("SCHOLAR_AGENT_WORK_ROOT", "D:/scholar agent")
    CODE_BUILDER_WORKSPACE_DIR = os.getenv(
        "CODE_BUILDER_WORKSPACE_DIR",
        os.path.join(SCHOLAR_AGENT_WORK_ROOT, "scholar code"),
    )
    CODE_BUILDER_PYTHON_EXECUTABLE = os.getenv("CODE_BUILDER_PYTHON_EXECUTABLE", sys.executable)
    WRITING_WORKSPACE_DIR = os.getenv(
        "WRITING_WORKSPACE_DIR",
        os.path.join(SCHOLAR_AGENT_WORK_ROOT, "scholar document"),
    )

    # ---- Web server ----
    WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
    WEB_PORT = _get_int_env("WEB_PORT", 8000)
    WEB_RELOAD = _get_bool_env("WEB_RELOAD", False)
    WEB_RELOAD_DIRS = os.getenv("WEB_RELOAD_DIRS", PROJECT_ROOT)

    # ---- Default / fallback values ----
    DEFAULT_PAPER_TITLE = os.getenv("DEFAULT_PAPER_TITLE", "Unknown")
    DEFAULT_LITERATURE_QUERY = os.getenv("DEFAULT_LITERATURE_QUERY", "research paper")
    DEFAULT_SUMMARIZER_QUERY = os.getenv("DEFAULT_SUMMARIZER_QUERY", "paper summary")
    DEFAULT_METHODOLOGY_QUERY_SUFFIX = os.getenv("DEFAULT_METHODOLOGY_QUERY_SUFFIX", "method algorithm architecture")

    # ---- External MCP client ----
    ENABLE_EXTERNAL_MCP = _get_bool_env("ENABLE_EXTERNAL_MCP", True)
    MCP_SERVERS_CONFIG = os.getenv("MCP_SERVERS_CONFIG", os.path.join(PROJECT_ROOT, "mcp_servers.yaml"))
    MCP_CONNECT_TIMEOUT = _get_int_env("MCP_CONNECT_TIMEOUT", 30)

    # ---- Skill system ----
    ENABLE_SKILLS = _get_bool_env("ENABLE_SKILLS", True)
    SKILLS_DIR = os.getenv("SKILLS_DIR", os.path.join(PROJECT_ROOT, "skills"))

    # ---- Agent memory ----
    ENABLE_AGENT_MEMORY = _get_bool_env("ENABLE_AGENT_MEMORY", True)
    AGENT_MEMORY_DB = os.getenv("AGENT_MEMORY_DB", os.path.join(DB_DIR, "agent_memory.db"))
    AGENT_MEMORY_SUMMARY_MAX_CHARS = _get_int_env("AGENT_MEMORY_SUMMARY_MAX_CHARS", 2400)
    AGENT_MEMORY_PROMPT_MAX_CHARS = _get_int_env("AGENT_MEMORY_PROMPT_MAX_CHARS", 1400)
    AGENT_MEMORY_MAX_TOPICS = _get_int_env("AGENT_MEMORY_MAX_TOPICS", 8)
    AGENT_MEMORY_MAX_PAPER_IDS = _get_int_env("AGENT_MEMORY_MAX_PAPER_IDS", 20)
    AGENT_MEMORY_MAX_EVENTS = _get_int_env("AGENT_MEMORY_MAX_EVENTS", 80)

    # ---- Logging ----
    LOG_DIR = os.getenv("SCHOLAR_AGENT_LOG_DIR", os.path.join(WORKSPACE_DIR, "logs"))
    LOG_FILE = os.getenv("SCHOLAR_AGENT_LOG_FILE", os.path.join(LOG_DIR, "scholar_agent.log"))
    LOG_LEVEL = os.getenv("SCHOLAR_AGENT_LOG_LEVEL", "INFO")
    LOG_TO_CONSOLE = _get_bool_env("SCHOLAR_AGENT_LOG_TO_CONSOLE", True)

    @classmethod
    def resolve_api_key(cls, agent_api_key: str, agent_base_url: str = "") -> str:
        """Return ``"not-needed"`` when the backend is Ollama, otherwise the real key."""
        if agent_api_key.lower() == "ollama":
            return "not-needed"
        if "ollama" in (agent_base_url or "").lower():
            return "not-needed"
        return agent_api_key

    @classmethod
    def check_config(cls):
        """Validate configuration and create required directories on startup."""
        os.makedirs(cls.DB_DIR, exist_ok=True)
        os.makedirs(cls.PAPERS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SCHOLAR_AGENT_WORK_ROOT, exist_ok=True)
        os.makedirs(cls.CODE_BUILDER_WORKSPACE_DIR, exist_ok=True)
        os.makedirs(cls.WRITING_WORKSPACE_DIR, exist_ok=True)
        logger.info(f"[config] project_root={cls.PROJECT_ROOT}")


conf = Config()

if __name__ == "__main__":
    conf.check_config()
