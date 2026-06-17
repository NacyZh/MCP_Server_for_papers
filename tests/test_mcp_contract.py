from rag import ScholarMCPServer


def test_registered_tool_contract_is_stable():
    server = ScholarMCPServer()

    assert server.get_registered_tool_names() == [
        "retrieve_evidence_chunks",
        "get_paper_outline",
        "get_paper_profile",
        "get_paper_summary",
        "build_paper_summary",
        "list_local_database",
        "search_local_database",
        "add_paper_to_database",
        "import_papers_from_directory",
        "get_tool_job_status",
        "rag_health_check",
        "delete_paper_from_database",
        "dedup_local_database",
        "backfill_paper_metadata",
        "evaluate_retrieval_quality",
    ]


def test_registered_tools_have_schemas():
    server = ScholarMCPServer()

    for tool in server.tools.values():
        schema = tool.get_mcp_schema()
        assert schema["name"] == tool.name
        assert schema["description"]
        assert schema["inputSchema"]["type"] == "object"
