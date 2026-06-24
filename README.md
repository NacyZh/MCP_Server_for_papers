# ScholarAgent RAG MCP Server

[English](README.md) | [简体中文](README_zh.md)

ScholarAgent is now a focused local-paper RAG MCP server. It exposes a small set of MCP tools for importing PDFs, indexing them into SQLite + ChromaDB, and retrieving section-aware paper chunks for Claude Code, Codex, or any MCP client.

## Features

- Local PDF ingestion into `PAPERS_DIR`.
- SQLite metadata, cached paper chunks, section outlines, and summary/profile caches.
- ChromaDB dense retrieval with BGE-M3 embeddings.
- BM25 keyword retrieval and reciprocal rank fusion.
- Lazy BGE reranker loading for searched chunks.
- Optional HyDE query expansion through an OpenAI-compatible API or local vLLM model.
- Retrieval quality evaluation with hybrid/dense/BM25 metric diffing.
- Background Qwen summary-cache construction for token-friendly paper overview tools.
- Structured tool errors plus input and path boundary checks.
- MCP stdio server for external coding assistants.

## Project Layout

```text
main.py                         MCP server entry point
config.py                       RAG/MCP configuration
rag/
  mcp_server.py                 FastMCP wrapper
  core/logging.py               production logging setup
  logs/                         runtime logs, ignored by Git
  evaluation/
    data/                       retrieval evaluation datasets
    results/                    evaluation reports, ignored by Git
    run_retrieval_eval.py       command-line retrieval evaluation
  plugins/
    pdf_parser.py               PDF parsing
    hyde.py                     optional HyDE query expansion
  storage/
    sqlite_store.py             paper metadata and chunks
    vector_store.py             ChromaDB + BGE + BM25 retrieval
    paper_manager.py            ingestion and retrieval orchestration
  tools/
    base.py                     tool base classes
    paper_tools.py              RAG MCP tools
```

## Requirements

- Python 3.11.
- Local or downloadable BGE models:
  - `BAAI/bge-m3`
  - `BAAI/bge-reranker-v2-m3`
- For PDF parsing, install the dependencies in `requirements.txt`.
- CUDA is recommended for large local models, but CPU works for small tests and metadata operations.
- For local generated summaries, install the optional `vllm` extra and place or download Qwen under `rag/models`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust paths if needed:

```text
PAPERS_DIR=path/to/papers
BGE_M3_MODEL_PATH=./rag/models/bge-m3
BGE_RERANKER_MODEL_PATH=./rag/models/bge-reranker-v2-m3
BGE_AUTO_DOWNLOAD=true
BGE_OFFLINE_MODE=false
BGE_M3_MODEL_REVISION=main
BGE_RERANKER_MODEL_REVISION=main
PAPER_PARSER_DEVICE=cuda
RAG_SUMMARY_BACKEND=api
RAG_SUMMARY_API_BASE_URL=http://127.0.0.1:8001/v1
RAG_SUMMARY_API_MODEL=Qwen/Qwen3-8B-AWQ
RAG_SUMMARY_MODEL_NAME=Qwen/Qwen3-8B-AWQ
RAG_SUMMARY_MODEL_PATH=./rag/models/Qwen3-8B-AWQ
```

If `BGE_AUTO_DOWNLOAD=true`, missing BGE models are downloaded from Hugging Face into `rag/models`. Downloads use a per-model lock, a temporary download directory, and a completion marker so concurrent MCP processes do not corrupt partially downloaded models. In offline environments, place complete models at the configured paths and set `BGE_OFFLINE_MODE=true`.

## MCP Tools

The server exposes only local RAG/database tools:

```text
retrieve_evidence_chunks
get_paper_outline
get_paper_profile
get_paper_summary
build_paper_summary
list_local_database
search_local_database
add_paper_to_database
import_papers_from_directory
get_tool_job_status
rag_health_check
delete_paper_from_database
dedup_local_database
backfill_paper_metadata
evaluate_retrieval_quality
```

Use the tools by intent:

- `get_paper_outline`: inspect paper sections and whether cached section summaries exist. It returns metadata only, not body text.
- `build_paper_summary`: submit a background job to build cached profile, paper summaries, and section summaries with the configured local Qwen summary model. If the model is missing, offline, or the selected backend is unavailable, the job fails with a structured summary-model error.
- `get_paper_profile`: retrieve a compact cached profile for relevance checks, planning, and multi-paper comparison.
- `get_paper_summary`: retrieve a cached structured summary for report drafting or method overview.
- `retrieve_evidence_chunks`: retrieve exact supporting passages for citations, verification, and quote-level evidence.

`retrieve_evidence_chunks` supports:

- `query`: semantic + keyword search over local paper chunks.
- `paper_ids`: comma-separated local IDs for direct paper reading.
- `section`: optional section filter, such as `Introduction`, `Method`, or `Experiments`.
- `top_k`: number of chunks to return.
- `max_chars`: total evidence text budget.

Import tools submit `build_paper_summary` jobs automatically after successful PDF imports. Pass `build_summary=false` to skip summary cache generation for a specific import.

Summary/profile tools only read precomputed cache. If the cache is missing they return a structured `SUMMARY_NOT_READY` error with `suggested_tool=build_paper_summary`; they do not synchronously summarize full papers inside the request.

## Summary Model

Generated summary caching is controlled by `RAG_SUMMARY_*` settings. The default model is:

```text
RAG_SUMMARY_MODEL_NAME=Qwen/Qwen3-8B-AWQ
RAG_SUMMARY_BACKEND=api
RAG_SUMMARY_API_BASE_URL=http://127.0.0.1:8001/v1
RAG_SUMMARY_API_KEY=
RAG_SUMMARY_API_MODEL=Qwen/Qwen3-8B-AWQ
RAG_SUMMARY_API_TIMEOUT_SECONDS=180
RAG_SUMMARY_API_MAX_RETRIES=2
RAG_SUMMARY_MODEL_PATH=./rag/models/Qwen3-8B-AWQ
RAG_SUMMARY_MODEL_REPO=Qwen/Qwen3-8B-AWQ
RAG_SUMMARY_MODEL_REVISION=main
RAG_SUMMARY_AUTO_DOWNLOAD=true
RAG_SUMMARY_OFFLINE_MODE=false
RAG_SUMMARY_GPU_MEMORY_UTILIZATION=0.75
RAG_SUMMARY_CPU_OFFLOAD_GB=0
RAG_SUMMARY_MAX_INPUT_TOKENS=4096
RAG_SUMMARY_MAX_OUTPUT_TOKENS=2048
RAG_SUMMARY_TEMPERATURE=0.2
RAG_SUMMARY_TOP_P=0.8
RAG_SUMMARY_CONCURRENCY=1
RAG_SUMMARY_ENFORCE_EAGER=true
```

`RAG_SUMMARY_BACKEND=api` calls an OpenAI-compatible `/v1/chat/completions` endpoint and does not load the summary model inside the MCP process. This is the recommended backend for native Windows and for running vLLM as a separate service. `RAG_SUMMARY_BACKEND=vllm` keeps the previous Linux/WSL path and lazy-loads the local model only inside `build_paper_summary` background jobs. MCP startup, metadata listing, database search, and evidence retrieval do not load Qwen.

`RAG_SUMMARY_CONCURRENCY` controls the dedicated `build_paper_summary` background queue. Keep it at `1` for single-GPU local services or API providers with strict rate/concurrency limits; increase it only when the backend can safely handle multiple simultaneous summary builds.

If `RAG_SUMMARY_BACKEND=vllm` and `RAG_SUMMARY_MODEL_PATH` is missing while automatic download is enabled, the model manager downloads `Qwen/Qwen3-8B-AWQ` into `rag/models/Qwen3-8B-AWQ` under the same process-safe lock pattern used for BGE models. If `RAG_SUMMARY_OFFLINE_MODE=true` or automatic download is disabled, a missing model is reported as `SUMMARY_MODEL_UNAVAILABLE`.

For API mode, run any OpenAI-compatible local or remote service and point `RAG_SUMMARY_API_BASE_URL` at its `/v1` base URL. Example vLLM service on Linux/WSL:

```bash
HF_HOME=/mnt/d/scholar-agent/.hf-cache \
.venv/bin/vllm serve /mnt/d/scholar-agent/rag/models/Qwen3-8B-AWQ \
  --served-model-name Qwen/Qwen3-8B-AWQ \
  --host 127.0.0.1 \
  --port 8001 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --enforce-eager
```

Install vLLM only when using the in-process Linux/WSL backend:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[summary-vllm]"
```

The `summary-vllm` extra is enabled only on non-Windows platforms. Native Windows should use `RAG_SUMMARY_BACKEND=api`.

For an already downloaded local model, either copy it to `rag/models/Qwen3-8B-AWQ` or point `RAG_SUMMARY_MODEL_PATH` at the actual model directory under `rag/models`. To use a different model, set `RAG_SUMMARY_MODEL_NAME`, `RAG_SUMMARY_MODEL_REPO`, and `RAG_SUMMARY_MODEL_PATH` together, for example `Qwen/Qwen3-4B` with `./rag/models/Qwen3-4B`, or `Qwen/Qwen3-14B` with `./rag/models/Qwen3-14B`.

## Run

List registered tools:

```powershell
.\.venv\Scripts\python.exe main.py --mode list-tools
```

Start stdio MCP server:

```powershell
.\.venv\Scripts\python.exe main.py --mode server --transport stdio
```

Claude Code example:

```powershell
claude mcp add --transport stdio --scope user scholaragent -- <PROJECT_ROOT>/.venv/Scripts/python.exe <PROJECT_ROOT>/main.py --mode server --transport stdio
```

Codex CLI example:

```toml
[mcp_servers.scholaragent]
command = "<PROJECT_ROOT>/.venv/Scripts/python.exe"
args = ["<PROJECT_ROOT>/main.py", "--mode", "server", "--transport", "stdio"]
cwd = "<PROJECT_ROOT>"
startup_timeout_sec = 30
tool_timeout_sec = 120
```

## Import And Retrieve

Place PDFs in `PAPERS_DIR`, then call `add_paper_to_database` with the filename:

```json
{
  "filename": "paper.pdf",
  "on_duplicate": "skip"
}
```

PDF import runs as a background job. The tool returns a `job_id`; call `get_tool_job_status` until the status is `succeeded` or `failed`.

`get_tool_job_status` long-polls active jobs instead of returning immediately. For `queued` or `running` jobs, each repeated status check waits with an arithmetic backoff controlled by `TOOL_JOB_STATUS_WAIT_INITIAL_SECONDS`, `TOOL_JOB_STATUS_WAIT_STEP_SECONDS`, and `TOOL_JOB_STATUS_WAIT_MAX_SECONDS` before returning. Completed or failed jobs return immediately.

To import every PDF in `PAPERS_DIR`, use `import_papers_from_directory`:

```json
{
  "subdir": "",
  "recursive": false,
  "on_duplicate": "skip",
  "max_files": 200,
  "dry_run": false
}
```

`subdir` is optional and must be relative to `PAPERS_DIR`. The batch job continues when an individual PDF fails and returns per-file `imported`, `skipped`, or `failed` results in the job payload.

Use `rag_health_check` for lightweight operational visibility into configured paths, model presence, and background job counts. Tool and job results include request IDs and elapsed time in structured payloads; server logs also record matching start/end events.

Runtime logs are written to `rag/logs/rag.log` by default. Logs use rotation, UTF-8 encoding, conservative console output for stdio MCP, and best-effort redaction for common secret fields such as API keys and tokens. `rag/logs/` is runtime state and is ignored by Git.

Tool failures are returned with structured error metadata:

```json
{
  "status": "fail",
  "error_code": "PDF_NOT_FOUND",
  "recoverable": true,
  "suggestion": "Copy the PDF into PAPERS_DIR and retry with the exact filename."
}
```

Search local content:

```json
{
  "query": "MIMO OFDM interference rejection receiver",
  "top_k": 5
}
```

Read a known paper section:

```json
{
  "paper_ids": "local_9025dd7c",
  "section": "Introduction",
  "top_k": 3
}
```

Build and read a cached summary:

```json
{
  "paper_id": "local_9025dd7c",
  "language": "en",
  "detail_levels": ["short", "medium", "long"],
  "force_rebuild": false
}
```

After `get_tool_job_status` reports success:

```json
{
  "paper_id": "local_9025dd7c",
  "detail_level": "medium",
  "language": "en",
  "include_sections": true,
  "max_chars": 12000
}
```

## Retrieval Evaluation

`evaluate_retrieval_quality` runs as a background job and compares retrieval modes on a JSONL dataset. By default it reads:

```text
RETRIEVAL_EVAL_DATASET_PATH=./rag/evaluation/data/retrieval_eval.jsonl
RETRIEVAL_EVAL_RESULTS_DIR=./rag/evaluation/results
```

Each row must contain `query` and at least one expected field:

```json
{"query":"attention mechanism","expected_paper_ids":["local_9025dd7c"],"expected_sections":["Method"]}
```

The job reports `paper_recall@k`, `chunk_recall@k`, `section_hit@k`, `precision@k`, `mrr`, and metric deltas between modes such as `hybrid`, `dense`, and `bm25`.

For real-environment evaluation, first create `rag/evaluation/data/retrieval_eval.jsonl` from `rag/evaluation/data/retrieval_eval.sample.jsonl` and replace `local_replace_me` with IDs returned by `list_local_database` or `search_local_database`. Then run:

```powershell
.\.venv\Scripts\python.exe -m rag.evaluation.run_retrieval_eval --dataset rag/evaluation/data/retrieval_eval.jsonl --modes hybrid,dense,bm25 --top-k 10 --max-cases 200
```

The script writes a full JSON report to `rag/evaluation/results/retrieval_eval_<timestamp>.json` and prints the aggregate summary. Evaluation datasets and result reports are local operational data; only the sample JSONL is committed.

## Security Boundaries

MCP tools intentionally accept a narrow input surface:

- `add_paper_to_database.filename` must be a plain PDF filename directly under `PAPERS_DIR`; absolute paths, nested paths, and traversal fragments are rejected.
- `import_papers_from_directory.subdir` must stay inside `PAPERS_DIR`; use `dry_run=true` to preview matched PDFs before importing.
- `paper_ids`, `local_id`, and `job_id` are format-validated before storage operations run.
- Query, section, paper ID count, evaluation case count, and returned text size are bounded by `.env` settings such as `TOOL_MAX_QUERY_CHARS` and `TOOL_MAX_RETURN_CHARS`.
- Evaluation dataset paths must stay inside the project root and use `.jsonl`.

## HyDE

HyDE is optional and controlled by:

```text
ENABLE_HYDE=true
HYDE_BACKEND=api
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL_NAME=
HYDE_API_TIMEOUT=60
```

For local vLLM inference:

```text
HYDE_BACKEND=vllm
HYDE_VLLM_MODEL_PATH=./rag/finetune/models/hyde-qwen-lora
```

Optional training utilities are in `scripts/` and can generate LLaMA-Factory configs for HyDE fine-tuning.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check rag main.py config.py tests
.\.venv\Scripts\python.exe -m mypy rag main.py config.py --no-sqlite-cache
.\.venv\Scripts\python.exe main.py --mode list-tools
.\.venv\Scripts\python.exe -m build --no-isolation
.\.venv\Scripts\python.exe -m pip_audit --local --cache-dir .pip-audit-cache --progress-spinner off
```
