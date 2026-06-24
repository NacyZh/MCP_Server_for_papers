import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_papers_dir_can_be_configured_independently(monkeypatch, tmp_path):
    import config as config_mod

    with monkeypatch.context() as env:
        env.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
        env.setenv("PAPERS_DIR", str(tmp_path / "custom_papers"))
        reloaded = importlib.reload(config_mod)

        assert Path(reloaded.conf.WORKSPACE_DIR) == tmp_path / "workspace"
        assert Path(reloaded.conf.PAPERS_DIR) == tmp_path / "custom_papers"
        assert Path(reloaded.conf.MODELS_DIR).name == "models"
        assert Path(reloaded.conf.MODELS_DIR).parent.name == "rag"

    importlib.reload(config_mod)


def test_default_logs_and_evaluation_paths_are_under_rag():
    import config as config_mod

    cfg = config_mod.Config(WORKSPACE_DIR="./workspace")

    assert Path(cfg.LOG_DIR).parts[-2:] == ("rag", "logs")
    assert Path(cfg.LOG_FILE).parts[-3:] == ("rag", "logs", "rag.log")
    assert Path(cfg.RETRIEVAL_EVAL_DATASET_PATH).parts[-4:] == (
        "rag",
        "evaluation",
        "data",
        "retrieval_eval.jsonl",
    )
    assert Path(cfg.RETRIEVAL_EVAL_RESULTS_DIR).parts[-3:] == ("rag", "evaluation", "results")


def test_config_validates_enums_and_ranges(monkeypatch):
    import config as config_mod

    with pytest.raises(ValidationError):
        config_mod.Config(PAPER_PARSER_DEVICE="gpu")

    with pytest.raises(ValidationError):
        config_mod.Config(HYDE_VLLM_GPU_MEMORY_UTILIZATION=1.5)


def test_env_example_keys_are_known_config_fields():
    import config as config_mod

    allowed = set(config_mod.Config.model_fields)
    aliases = {
        str(field.validation_alias)
        for field in config_mod.Config.model_fields.values()
        if field.validation_alias is not None
    }
    allowed.update(aliases)

    keys = []
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0].strip())

    unknown = sorted(set(keys) - allowed)
    assert unknown == []


def test_summary_api_config_inherits_common_llm_environment(monkeypatch):
    import config as config_mod

    with monkeypatch.context() as env:
        env.setenv("LLM_BASE_URL", " http://127.0.0.1:8001/v1 ")
        env.setenv("LLM_API_KEY", " local-key ")
        env.setenv("LLM_MODEL_NAME", " qwen-local ")
        env.delenv("RAG_SUMMARY_API_BASE_URL", raising=False)
        env.delenv("RAG_SUMMARY_API_KEY", raising=False)
        env.delenv("RAG_SUMMARY_API_MODEL", raising=False)
        reloaded = importlib.reload(config_mod)

        assert reloaded.conf.RAG_SUMMARY_API_BASE_URL == "http://127.0.0.1:8001/v1"
        assert reloaded.conf.RAG_SUMMARY_API_KEY == "local-key"
        assert reloaded.conf.RAG_SUMMARY_API_MODEL == "qwen-local"

    importlib.reload(config_mod)


def test_summary_api_config_explicit_env_overrides_common_llm_environment(monkeypatch):
    import config as config_mod

    with monkeypatch.context() as env:
        env.setenv("LLM_BASE_URL", "http://common.example/v1")
        env.setenv("LLM_API_KEY", "common-key")
        env.setenv("LLM_MODEL_NAME", "common-model")
        env.setenv("RAG_SUMMARY_API_BASE_URL", "http://summary.example/v1")
        env.setenv("RAG_SUMMARY_API_KEY", "summary-key")
        env.setenv("RAG_SUMMARY_API_MODEL", "summary-model")
        reloaded = importlib.reload(config_mod)

        assert reloaded.conf.RAG_SUMMARY_API_BASE_URL == "http://summary.example/v1"
        assert reloaded.conf.RAG_SUMMARY_API_KEY == "summary-key"
        assert reloaded.conf.RAG_SUMMARY_API_MODEL == "summary-model"

    importlib.reload(config_mod)


def test_paper_db_creates_missing_workspace_db_dir(monkeypatch, tmp_path):
    import config as config_mod
    import rag.storage.sqlite_store as sqlite_store

    monkeypatch.setattr(config_mod.conf, "DB_DIR", str(tmp_path / "missing_workspace" / "db"))
    monkeypatch.setattr(sqlite_store.conf, "DB_DIR", str(tmp_path / "missing_workspace" / "db"))

    db = sqlite_store.PaperDB()

    assert Path(db.db_path).exists()
    assert Path(db.db_path).parent == tmp_path / "missing_workspace" / "db"
    assert db.get_schema_version() == db.supported_schema_version


def test_paper_db_stores_and_deletes_chunks(monkeypatch, tmp_path):
    import config as config_mod
    import rag.storage.sqlite_store as sqlite_store

    monkeypatch.setattr(config_mod.conf, "DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(sqlite_store.conf, "DB_DIR", str(tmp_path / "db"))

    db = sqlite_store.PaperDB()
    db.add_paper("paper_1", "Demo")

    assert db.replace_paper_chunks(
        "paper_1",
        [
            {"content": "second", "section_name": "intro", "section_title": "Introduction"},
            {"content": "first", "section_name": "method", "section_title": "Method"},
        ],
    ) == 2
    rows = db.get_paper_chunks("paper_1", limit=1)
    assert [row["content"] for row in rows] == ["second"]
    assert rows[0]["section_title"] == "Introduction"
    assert db.get_paper_sections("paper_1") == ["Introduction (1)", "Method (1)"]

    db.delete_paper("paper_1")

    assert db.get_paper_chunks("paper_1") == []


def test_paper_db_metadata_search_matches_non_contiguous_terms(monkeypatch, tmp_path):
    import config as config_mod
    import rag.storage.sqlite_store as sqlite_store

    monkeypatch.setattr(config_mod.conf, "DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(sqlite_store.conf, "DB_DIR", str(tmp_path / "db"))

    db = sqlite_store.PaperDB()
    db.add_paper(
        "paper_transformer",
        "Attention Is All You Need",
        authors="Vaswani et al.",
        abstract="A sequence transduction model based on self attention.",
        tags="transformer",
        normalized_title="attention is all you need",
    )
    db.add_paper(
        "paper_graph",
        "Graph Retrieval for RAG",
        authors="Demo",
        abstract="Graph neural retrieval for local paper search.",
        tags="rag graph",
        normalized_title="graph retrieval for rag",
    )

    results = db.search_papers("transformer attention mechanism")

    assert results
    assert results[0]["id"] == "paper_transformer"
    assert {item["id"] for item in results} == {"paper_transformer"}


def test_summary_cache_key_includes_model_and_prompt(monkeypatch, tmp_path):
    import config as config_mod
    import rag.storage.sqlite_store as sqlite_store

    class FakeSummaryGenerator:
        model_name = "Qwen/Qwen3-8B"
        prompt_version = "qwen3-summary-v1"

        def summarize_section(self, **kwargs):
            return "generated section summary"

        def build_profile(self, **kwargs):
            return {
                "problem": "p",
                "background": "b",
                "method": "m",
                "contributions": ["c"],
                "experiments": "e",
                "limitations": "l",
                "keywords": ["k"],
            }

        def build_paper_summary(self, **kwargs):
            return {
                "background": "b",
                "problem": "p",
                "method": "m",
                "experiments": "e",
                "results": "r",
                "limitations": "l",
                "takeaways": "t",
            }

    class NewPromptSummaryGenerator(FakeSummaryGenerator):
        prompt_version = "qwen3-summary-v2"

    monkeypatch.setattr(config_mod.conf, "DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(sqlite_store.conf, "DB_DIR", str(tmp_path / "db"))

    db = sqlite_store.PaperDB()
    db.add_paper("paper_1", "Demo")
    db.replace_paper_chunks(
        "paper_1",
        [{"content": "method text", "section_name": "method", "section_title": "Method"}],
    )

    first = db.build_summary_cache(
        "paper_1",
        language="en",
        detail_levels=["medium"],
        summary_generator=FakeSummaryGenerator(),
    )
    assert first["detail_levels_built"] == ["medium"]

    second = db.build_summary_cache(
        "paper_1",
        language="en",
        detail_levels=["medium"],
        summary_generator=FakeSummaryGenerator(),
    )
    assert second["detail_levels_built"] == []
    assert second["detail_levels_skipped"] == ["medium"]

    third = db.build_summary_cache(
        "paper_1",
        language="en",
        detail_levels=["medium"],
        summary_generator=NewPromptSummaryGenerator(),
    )
    assert third["detail_levels_built"] == ["medium"]
    assert third["detail_levels_skipped"] == []


def test_summary_cache_reports_lazy_model_unavailable(monkeypatch, tmp_path):
    import config as config_mod
    import rag.storage.sqlite_store as sqlite_store
    from rag.plugins.summary_model import SummaryModelUnavailableError

    class UnavailableSummaryGenerator:
        model_name = "Qwen/Qwen3-8B"
        prompt_version = "qwen3-summary-v1"

        def summarize_section(self, **kwargs):
            raise SummaryModelUnavailableError("missing local summary model")

    monkeypatch.setattr(config_mod.conf, "DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(sqlite_store.conf, "DB_DIR", str(tmp_path / "db"))

    db = sqlite_store.PaperDB()
    db.add_paper("paper_1", "Demo")
    db.replace_paper_chunks(
        "paper_1",
        [{"content": "method text", "section_name": "method", "section_title": "Method"}],
    )

    report = db.build_summary_cache(
        "paper_1",
        language="en",
        detail_levels=["short"],
        summary_generator=UnavailableSummaryGenerator(),
    )

    assert report["status"] == "failed"
    assert report["error_code"] == "SUMMARY_MODEL_UNAVAILABLE"
