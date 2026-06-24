# ScholarAgent RAG MCP Server

[English](README.md) | [简体中文](README_zh.md)

ScholarAgent 现在是一个聚焦本地论文库的 RAG MCP Server。它暴露一组精简的 MCP 工具，用于导入 PDF、写入 SQLite + ChromaDB 索引，并为 Claude Code、Codex 或任意 MCP 客户端提供按章节感知的论文片段检索能力。

## 功能特性

- 将本地 PDF 导入到 `PAPERS_DIR`。
- 使用 SQLite 存储论文元数据、缓存论文分块、章节大纲、摘要和 profile。
- 使用 BGE-M3 embedding 进行 ChromaDB dense retrieval。
- 支持 BM25 关键词检索和 RRF 融合。
- BGE reranker 按需懒加载，只在检索重排时加载。
- 可选 HyDE 查询扩展，支持 OpenAI-compatible API 或本地 vLLM 模型。
- 支持 retrieval quality evaluation，对 hybrid、dense、BM25 等模式进行指标对比。
- 后台构建 Qwen 摘要缓存，为论文概览工具节省 token。
- 工具错误结构化返回，并限制输入、路径和返回文本边界。
- 通过 stdio MCP server 暴露给外部 coding assistant。

## 项目结构

```text
main.py                         MCP server 入口
config.py                       RAG/MCP 配置
rag/
  mcp_server.py                 FastMCP 封装
  core/logging.py               生产级日志配置
  logs/                         运行日志，Git 忽略
  evaluation/
    data/                       retrieval evaluation 数据集
    results/                    evaluation 报告，Git 忽略
    run_retrieval_eval.py       命令行 retrieval evaluation
  plugins/
    pdf_parser.py               PDF 解析
    hyde.py                     可选 HyDE 查询扩展
  storage/
    sqlite_store.py             论文元数据和分块
    vector_store.py             ChromaDB + BGE + BM25 检索
    paper_manager.py            导入和检索编排
  tools/
    base.py                     工具基类
    paper_tools.py              RAG MCP 工具
```

## 环境要求

- Python 3.11。
- 本地已有或可自动下载的 BGE 模型：
  - `BAAI/bge-m3`
  - `BAAI/bge-reranker-v2-m3`
- PDF 解析需要安装 `requirements.txt` 中的依赖。
- 大模型本地推理推荐 CUDA；CPU 可用于小规模测试和元数据操作。
- 如需本地生成摘要，安装可选的 `vllm` extra，并将 Qwen 模型放置或下载到 `rag/models`。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并按需修改路径：

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

如果 `BGE_AUTO_DOWNLOAD=true`，缺失的 BGE 模型会从 Hugging Face 下载到 `rag/models`。下载使用每个模型独立的锁、临时下载目录和完成标记，避免多个 MCP 进程同时启动时破坏未完成的模型目录。离线环境中，请提前将完整模型放到配置路径，并设置 `BGE_OFFLINE_MODE=true`。

## MCP 工具

服务只暴露本地 RAG/database 工具：

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

按用途选择工具：

- `get_paper_outline`：查看论文章节和章节摘要缓存状态。只返回元数据，不返回正文。
- `build_paper_summary`：提交后台任务，使用配置的 Qwen 摘要模型构建 profile、论文摘要和章节摘要缓存。模型缺失、离线或后端不可用时，任务会返回结构化 summary-model 错误。
- `get_paper_profile`：读取紧凑的缓存 profile，用于相关性判断、规划和多论文比较。
- `get_paper_summary`：读取缓存的结构化摘要，用于报告草稿或方法概览。
- `retrieve_evidence_chunks`：检索可用于引用、核验和证据定位的精确文本片段。

`retrieve_evidence_chunks` 支持：

- `query`：在本地论文分块上做语义 + 关键词检索。
- `paper_ids`：逗号分隔的本地论文 ID，用于直接读取指定论文。
- `section`：可选章节过滤，例如 `Introduction`、`Method`、`Experiments`。
- `top_k`：返回片段数量。
- `max_chars`：返回证据文本的总字符预算。

导入工具会在 PDF 成功导入后自动提交 `build_paper_summary` 任务。对单次导入传入 `build_summary=false` 可以跳过摘要缓存生成。

summary/profile 工具只读取预计算缓存。如果缓存不存在，会返回结构化 `SUMMARY_NOT_READY` 错误，并带有 `suggested_tool=build_paper_summary`；它们不会在请求中同步总结整篇论文。

## 摘要模型

摘要缓存由 `RAG_SUMMARY_*` 配置控制。默认模型为：

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

`RAG_SUMMARY_BACKEND=api` 会调用 OpenAI-compatible `/v1/chat/completions` 接口，不在 MCP 进程内加载摘要模型。这是原生 Windows 和“vLLM 作为独立服务运行”场景的推荐方式。`RAG_SUMMARY_BACKEND=vllm` 保留 Linux/WSL 的进程内本地推理路径，只在 `build_paper_summary` 后台任务中懒加载本地模型。MCP 启动、元数据列表、数据库搜索和证据检索都不会加载 Qwen。

`RAG_SUMMARY_CONCURRENCY` 控制专用的 `build_paper_summary` 后台队列。单 GPU 本地服务或 API 提供方有严格并发限制时建议保持为 `1`；只有后端可以安全处理多个并发 summary build 时再提高。

如果 `RAG_SUMMARY_BACKEND=vllm`，且 `RAG_SUMMARY_MODEL_PATH` 缺失并启用了自动下载，模型管理器会使用和 BGE 模型相同的进程安全锁机制，将 `Qwen/Qwen3-8B-AWQ` 下载到 `rag/models/Qwen3-8B-AWQ`。如果 `RAG_SUMMARY_OFFLINE_MODE=true` 或关闭自动下载，缺失模型会报告为 `SUMMARY_MODEL_UNAVAILABLE`。

API 模式下，启动任意 OpenAI-compatible 本地或远程服务，并将 `RAG_SUMMARY_API_BASE_URL` 指向其 `/v1` base URL。Linux/WSL 上的 vLLM 服务示例：

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

只有使用 Linux/WSL 进程内后端时才需要安装 vLLM：

```powershell
.\.venv\Scripts\python.exe -m pip install ".[summary-vllm]"
```

`summary-vllm` extra 只在非 Windows 平台启用。原生 Windows 应使用 `RAG_SUMMARY_BACKEND=api`。

对于已下载的本地模型，可以复制到 `rag/models/Qwen3-8B-AWQ`，或将 `RAG_SUMMARY_MODEL_PATH` 指向 `rag/models` 下的实际模型目录。使用其它模型时，需要同时设置 `RAG_SUMMARY_MODEL_NAME`、`RAG_SUMMARY_MODEL_REPO` 和 `RAG_SUMMARY_MODEL_PATH`，例如 `Qwen/Qwen3-4B` + `./rag/models/Qwen3-4B`，或 `Qwen/Qwen3-14B` + `./rag/models/Qwen3-14B`。

## 运行

列出已注册工具：

```powershell
.\.venv\Scripts\python.exe main.py --mode list-tools
```

启动 stdio MCP server：

```powershell
.\.venv\Scripts\python.exe main.py --mode server --transport stdio
```

Claude Code 示例：

```powershell
claude mcp add --transport stdio --scope user scholaragent -- <PROJECT_ROOT>/.venv/Scripts/python.exe <PROJECT_ROOT>/main.py --mode server --transport stdio
```

Codex CLI 示例：

```toml
[mcp_servers.scholaragent]
command = "<PROJECT_ROOT>/.venv/Scripts/python.exe"
args = ["<PROJECT_ROOT>/main.py", "--mode", "server", "--transport", "stdio"]
cwd = "<PROJECT_ROOT>"
startup_timeout_sec = 30
tool_timeout_sec = 120
```

## 导入与检索

将 PDF 放入 `PAPERS_DIR`，然后调用 `add_paper_to_database` 并传入文件名：

```json
{
  "filename": "paper.pdf",
  "on_duplicate": "skip"
}
```

PDF 导入会作为后台任务运行。工具会返回 `job_id`；调用 `get_tool_job_status`，直到状态变为 `succeeded` 或 `failed`。

`get_tool_job_status` 会对活跃任务进行长轮询，而不是立即返回。对于 `queued` 或 `running` 的任务，重复状态检查会按等差递增等待，等待时间由 `TOOL_JOB_STATUS_WAIT_INITIAL_SECONDS`、`TOOL_JOB_STATUS_WAIT_STEP_SECONDS` 和 `TOOL_JOB_STATUS_WAIT_MAX_SECONDS` 控制。已完成或失败的任务会立即返回。

导入 `PAPERS_DIR` 中的所有 PDF，使用 `import_papers_from_directory`：

```json
{
  "subdir": "",
  "recursive": false,
  "on_duplicate": "skip",
  "max_files": 200,
  "dry_run": false
}
```

`subdir` 是可选参数，必须是 `PAPERS_DIR` 内部的相对路径。批量任务在单个 PDF 失败时会继续处理，并在任务 payload 中返回每个文件的 `imported`、`skipped` 或 `failed` 结果。

使用 `rag_health_check` 可以轻量查看配置路径、模型存在性和后台 job 计数。工具和 job 结果会包含 request ID 和耗时；服务日志也会记录匹配的开始/结束事件。

运行日志默认写入 `rag/logs/rag.log`。日志启用轮转、UTF-8 编码、适合 stdio MCP 的保守控制台输出，并尽力脱敏 API key、token 等常见敏感字段。`rag/logs/` 是运行态数据，Git 会忽略。

工具失败会返回结构化错误元数据：

```json
{
  "status": "fail",
  "error_code": "PDF_NOT_FOUND",
  "recoverable": true,
  "suggestion": "Copy the PDF into PAPERS_DIR and retry with the exact filename."
}
```

搜索本地内容：

```json
{
  "query": "MIMO OFDM interference rejection receiver",
  "top_k": 5
}
```

读取已知论文的指定章节：

```json
{
  "paper_ids": "local_9025dd7c",
  "section": "Introduction",
  "top_k": 3
}
```

构建并读取缓存摘要：

```json
{
  "paper_id": "local_9025dd7c",
  "language": "en",
  "detail_levels": ["short", "medium", "long"],
  "force_rebuild": false
}
```

等待 `get_tool_job_status` 报告成功后：

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

`evaluate_retrieval_quality` 会作为后台任务运行，并在 JSONL 数据集上对比不同检索模式。默认读取：

```text
RETRIEVAL_EVAL_DATASET_PATH=./rag/evaluation/data/retrieval_eval.jsonl
RETRIEVAL_EVAL_RESULTS_DIR=./rag/evaluation/results
```

每一行必须包含 `query`，并至少包含一个 expected 字段：

```json
{"query":"attention mechanism","expected_paper_ids":["local_9025dd7c"],"expected_sections":["Method"]}
```

任务会报告 `paper_recall@k`、`chunk_recall@k`、`section_hit@k`、`precision@k`、`mrr`，以及 `hybrid`、`dense`、`bm25` 等模式之间的指标差异。

真实环境评估时，先从 `rag/evaluation/data/retrieval_eval.sample.jsonl` 创建 `rag/evaluation/data/retrieval_eval.jsonl`，并将 `local_replace_me` 替换为 `list_local_database` 或 `search_local_database` 返回的 ID。然后运行：

```powershell
.\.venv\Scripts\python.exe -m rag.evaluation.run_retrieval_eval --dataset rag/evaluation/data/retrieval_eval.jsonl --modes hybrid,dense,bm25 --top-k 10 --max-cases 200
```

脚本会将完整 JSON 报告写入 `rag/evaluation/results/retrieval_eval_<timestamp>.json`，并打印聚合摘要。Evaluation 数据集和结果报告属于本地运行数据；仓库只提交 sample JSONL。

## 安全边界

MCP 工具有意保持较窄的输入面：

- `add_paper_to_database.filename` 必须是 `PAPERS_DIR` 直属的普通 PDF 文件名；拒绝绝对路径、嵌套路径和路径穿越片段。
- `import_papers_from_directory.subdir` 必须停留在 `PAPERS_DIR` 内；导入前可用 `dry_run=true` 预览匹配的 PDF。
- `paper_ids`、`local_id`、`job_id` 在执行存储操作前都会进行格式校验。
- 查询、章节、论文 ID 数量、evaluation case 数量和返回文本大小都受 `.env` 中的 `TOOL_MAX_QUERY_CHARS`、`TOOL_MAX_RETURN_CHARS` 等配置限制。
- Evaluation dataset 路径必须位于项目根目录内，并使用 `.jsonl`。

## HyDE

HyDE 是可选功能，由以下配置控制：

```text
ENABLE_HYDE=true
HYDE_BACKEND=api
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL_NAME=
HYDE_API_TIMEOUT=60
```

本地 vLLM 推理配置：

```text
HYDE_BACKEND=vllm
HYDE_VLLM_MODEL_PATH=./rag/finetune/models/hyde-qwen-lora
```

可选训练工具位于 `scripts/`，可生成 HyDE 微调使用的 LLaMA-Factory 配置。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

检查：

```powershell
.\.venv\Scripts\python.exe -m ruff check rag main.py config.py tests
.\.venv\Scripts\python.exe -m mypy rag main.py config.py --no-sqlite-cache
.\.venv\Scripts\python.exe main.py --mode list-tools
.\.venv\Scripts\python.exe -m build --no-isolation
.\.venv\Scripts\python.exe -m pip_audit --local --cache-dir .pip-audit-cache --progress-spinner off
```
