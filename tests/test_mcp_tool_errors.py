import json

from rag.mcp_server import ScholarMCPServer


def test_mcp_failure_text_contains_structured_error(monkeypatch):
    server = ScholarMCPServer()

    text = server._execute_tool("search_local_database")

    assert text.startswith("[tool_error] ")
    payload = json.loads(text.removeprefix("[tool_error] "))
    assert payload["status"] == "fail"
    assert payload["error_code"] == "MISSING_REQUIRED_ARGUMENT"
    assert payload["request_id"].startswith("tool_")
    assert isinstance(payload["elapsed_ms"], int)


def test_mcp_unknown_tool_returns_structured_error():
    server = ScholarMCPServer()

    text = server._execute_tool("missing_tool")

    payload = json.loads(text.removeprefix("[tool_error] "))
    assert payload["error_code"] == "TOOL_NOT_FOUND"
