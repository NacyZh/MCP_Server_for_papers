import sys

from scholar_agent.tools.paper_tools import DbListTool, DbSearchTool, build_default_tools


class FakePaperDB:
    def get_all_papers(self):
        return [{"id": "local_1", "title": "Demo Paper"}]

    def search_papers(self, keyword):
        assert keyword == "attention"
        return [{
            "id": "local_2",
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "publish_year": 2017,
            "tags": "transformer",
        }]


def test_list_local_database_does_not_import_vector_store(monkeypatch):
    sys.modules.pop("scholar_agent.storage.vector_store", None)
    sys.modules.pop("scholar_agent.plugins.pdf_parser", None)

    result = DbListTool(paper_db=FakePaperDB()).execute()

    assert result.status == "success"
    assert "local_1" in result.result
    assert "scholar_agent.storage.vector_store" not in sys.modules
    assert "scholar_agent.plugins.pdf_parser" not in sys.modules


def test_search_local_database_does_not_import_vector_store():
    sys.modules.pop("scholar_agent.storage.vector_store", None)
    sys.modules.pop("scholar_agent.plugins.pdf_parser", None)

    result = DbSearchTool(paper_db=FakePaperDB()).execute(query="attention")

    assert result.status == "success"
    assert "local_2" in result.result
    assert "Attention Is All You Need" in result.result
    assert result.data[0]["id"] == "local_2"
    assert "scholar_agent.storage.vector_store" not in sys.modules
    assert "scholar_agent.plugins.pdf_parser" not in sys.modules


def test_default_tools_include_local_database_list_and_search(monkeypatch):
    monkeypatch.setattr(
        "scholar_agent.mcp_client.discover_external_tools",
        lambda: None,
    )

    tools = build_default_tools()

    assert "list_local_database" in tools
    assert "search_local_database" in tools
    assert "search_local_papers_chunks" in tools
    assert "get_local_paper_chunks" in tools
