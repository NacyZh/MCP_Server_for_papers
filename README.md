# ScholarAgent

ScholarAgent is a local academic research assistant for literature search, paper summarization, methodology analysis, code reproduction, and academic writing. It combines a LangGraph multi-agent workflow, local paper retrieval, arXiv search, controlled code/document workspaces, a FastAPI web UI, and an MCP server.

The project is designed for local-first research workflows: papers, vector indexes, logs, generated code, and generated documents stay on your machine unless you explicitly connect external services.

## Features

- Multi-agent workflow: Supervisor, Literature, Summarizer, Methodology, Code Builder, Writing Editor, Database Manager, and Synthesis.
- Local paper database: import PDFs, store metadata in SQLite, index chunks in ChromaDB, deduplicate, delete, and backfill metadata.
- Hybrid retrieval: BGE-M3 dense embeddings, BM25 sparse retrieval, reciprocal-rank fusion, and BGE reranking.
- arXiv integration: search and download papers through the arXiv API.
- Code Builder: creates or edits project files in a controlled workspace, runs Python, pytest, shell commands, and external MCP tools.
- Writing Editor: reads and writes `.docx`, `.tex`, `.bib`, `.md`, and `.txt`; can compile LaTeX through local TeX tools.
- Web UI: multi-turn chat, conversation history, markdown rendering, live agent status, workspace picker, and Python interpreter picker.
- MCP server: exposes ScholarAgent tools to MCP-compatible clients.

## Architecture

The main graph is intentionally simple:

```text
START -> supervisor -> module_executor -> synthesis -> END
```

The Supervisor asks the LLM for a structured plan. The Module Executor runs the selected expert modules once in order. Synthesis produces the final answer. The code avoids keyword-table routing for user intent; the LLM plans, while code validates schema, paths, limits, and tool results.

Key modules:

```text
scholar_agent/
  agents/             LangGraph state, graph, service, supervisor, executor, synthesis
  agents/experts/     Literature, summarizer, methodology, code builder, database manager, writing editor
  tools/              Local paper tools, code workspace tools, writing tools, registry
  storage/            SQLite, ChromaDB, paper manager, memory store
  plugins/            arXiv, PDF parser, HyDE
  web/                FastAPI routes and static frontend
  skills/             Skill loader and skill model
main.py               MCP server entry point
run_web.py            Web server entry point
```

## Requirements

- Python 3.11 or newer.
- A working LLM-compatible API endpoint. OpenAI-compatible services such as DeepSeek can be used through `LLM_BASE_URL`.
- Local BGE-M3 embedding model under `workspace/models/bge-m3`. This is required before importing PDFs because imported chunks are embedded into ChromaDB.
- Local BGE reranker under `workspace/models/bge-reranker-v2-m3`. This is required for reranked local search results.
- A CUDA-capable environment is recommended for `marker-pdf` and BGE models.
- Optional for LaTeX compilation: `latexmk`, `xelatex`, `lualatex`, or `pdflatex` on `PATH`.

## Installation

Create and activate an environment, then install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For development:

```powershell
python -m pip install -e ".[dev]"
```

## Local Models

Download the two BGE models before using local paper import or local search:

```text
workspace/models/bge-m3
workspace/models/bge-reranker-v2-m3
```

`bge-m3` is mandatory for `POST /api/papers/import` because PDF chunks are embedded during import. `bge-reranker-v2-m3` is used by local search reranking; without it, search falls back to the pre-rerank order.

If your models live elsewhere, set these paths in `.env`:

```text
BGE_M3_MODEL_PATH=D:/models/bge-m3
BGE_RERANKER_MODEL_PATH=D:/models/bge-reranker-v2-m3
PAPER_PARSER_DEVICE=auto
```

`PAPER_PARSER_DEVICE=auto` uses CUDA when available and falls back to CPU otherwise. Set it to `cuda` only when the active Python environment has a CUDA-enabled PyTorch build.

## Configuration

Copy `.env.example` to `.env` and edit the values:

```powershell
Copy-Item .env.example .env
```

Minimum LLM configuration:

```text
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat
```

Useful runtime paths:

```text
WORKSPACE_DIR=./workspace
SCHOLAR_AGENT_WORK_ROOT=D:/scholar agent
CODE_BUILDER_WORKSPACE_DIR=D:/scholar agent/scholar code
CODE_BUILDER_PYTHON_EXECUTABLE=D:/Anaconda/envs/scholaragent311/python.exe
WRITING_WORKSPACE_DIR=D:/scholar agent/scholar document
```

The web UI also lets you select the Code Builder workspace and Python interpreter path per request.

## Run The Web UI

```powershell
python run_web.py
```

Default URL:

```text
http://127.0.0.1:8000
```

For development reload:

```text
WEB_RELOAD=true
WEB_RELOAD_DIRS=.
```

## Run The MCP Server

List tools:

```powershell
python main.py --mode list-tools
```

Start the MCP server over stdio:

```powershell
python main.py --mode server --transport stdio
```

External MCP tools can be configured in `mcp_servers.yaml`. Disable discovery when you only want local smoke tests:

```powershell
$env:ENABLE_EXTERNAL_MCP='false'
```

## Web API

Common endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/tools` | List registered tools |
| `GET` | `/api/tools/schemas` | List tool schemas |
| `POST` | `/api/chat` | Blocking multi-agent chat |
| `POST` | `/api/chat/stream` | SSE streaming multi-agent chat |
| `GET` | `/api/chat/memory/{session_id}` | Inspect session memory |
| `DELETE` | `/api/chat/memory/{session_id}` | Clear session memory |
| `GET` | `/api/filesystem/roots` | Directory picker roots |
| `GET` | `/api/filesystem/directories` | Browse directories |
| `GET` | `/api/filesystem/python-files` | Browse Python interpreters |
| `GET` | `/api/local/database` | List local papers |
| `POST` | `/api/local/search` | Search local indexed papers |
| `POST` | `/api/papers/import` | Import a local PDF |
| `POST` | `/api/papers/delete` | Delete a local paper |
| `POST` | `/api/papers/dedup` | Detect or clean duplicates |
| `POST` | `/api/arxiv/search` | Search arXiv |
| `POST` | `/api/arxiv/download` | Download arXiv PDF |

## Workspaces

Runtime data is intentionally separated from source code:

```text
workspace/                         Local database, logs, papers, models
D:/scholar agent/scholar code       Code Builder output projects
D:/scholar agent/scholar document   Writing Editor output documents
```

Generated code and documents are controlled by path safety checks. Code tools operate under the selected project root and accept relative paths only.

## Skills

Skills are registered from the repository-root `skill.json`. The built-in example plugin uses `source: "./skills"`, so the `example` skill resolves to:

```text
skills/example/SKILL.md
```

To add another skill, create `skills/<skill-name>/SKILL.md` with YAML frontmatter and add `<skill-name>` to the root `skill.json` plugin `skills` list. Keep plugin paths relative to the repository root.

## Testing

Compile Python files:

```powershell
python -m compileall -q scholar_agent tests main.py run_web.py
```

Run workflow checks without real LLM/network calls:

```powershell
python tests/run_agent_workflow_checks.py
```

Run pytest:

```powershell
python -m pytest tests
```

Check frontend JavaScript syntax:

```powershell
node --check scholar_agent/web/static/app.js
```

## GitHub Notes

Do not commit local runtime data or secrets:

- `.env`
- `workspace/db/`
- `workspace/logs/`
- `workspace/models/`
- `workspace/papers/`
- generated external code/document folders outside this repository

The included `.gitignore` already excludes the repository-local workspace and common Python build artifacts.

## License

MIT
