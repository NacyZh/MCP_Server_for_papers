import importlib
from pathlib import Path


def test_papers_dir_can_be_configured_independently(monkeypatch, tmp_path):
    import scholar_agent.config as config_mod

    with monkeypatch.context() as env:
        env.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
        env.setenv("PAPERS_DIR", str(tmp_path / "custom_papers"))
        reloaded = importlib.reload(config_mod)

        assert Path(reloaded.conf.WORKSPACE_DIR) == tmp_path / "workspace"
        assert Path(reloaded.conf.PAPERS_DIR) == tmp_path / "custom_papers"
        assert Path(reloaded.conf.MODELS_DIR) == tmp_path / "workspace" / "models"

    importlib.reload(config_mod)


def test_paper_db_creates_missing_workspace_db_dir(monkeypatch, tmp_path):
    import scholar_agent.config as config_mod
    import scholar_agent.storage.sqlite_store as sqlite_store

    monkeypatch.setattr(config_mod.conf, "DB_DIR", str(tmp_path / "missing_workspace" / "db"))
    monkeypatch.setattr(sqlite_store.conf, "DB_DIR", str(tmp_path / "missing_workspace" / "db"))

    db = sqlite_store.PaperDB()

    assert Path(db.db_path).exists()
    assert Path(db.db_path).parent == tmp_path / "missing_workspace" / "db"
