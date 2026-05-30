from scholar_agent.mcp_client import MCPConnection, _expand_env_value


def test_mcp_config_expands_environment_variables(monkeypatch):
    monkeypatch.setenv("MATLAB_MCP_PYTHON_EXECUTABLE", "D:/Anaconda/envs/scholaragent311/python.exe")

    expanded = _expand_env_value(
        {
            "command": "${MATLAB_MCP_PYTHON_EXECUTABLE}",
            "args": ["--bin", "%MATLAB_MCP_PYTHON_EXECUTABLE%"],
            "env": {"MCP_BIN": "$MATLAB_MCP_PYTHON_EXECUTABLE"},
        }
    )

    assert expanded["command"] == "D:/Anaconda/envs/scholaragent311/python.exe"
    assert expanded["args"][1] == "D:/Anaconda/envs/scholaragent311/python.exe"
    assert expanded["env"]["MCP_BIN"] == "D:/Anaconda/envs/scholaragent311/python.exe"


def test_mcp_connection_uses_expanded_command(monkeypatch):
    monkeypatch.setenv("MATLAB_MCP_PYTHON_EXECUTABLE", "D:/Anaconda/envs/scholaragent311/python.exe")

    conn = MCPConnection("matlab", {"type": "stdio", "command": "${MATLAB_MCP_PYTHON_EXECUTABLE}", "args": []})

    assert conn.command == "D:/Anaconda/envs/scholaragent311/python.exe"
