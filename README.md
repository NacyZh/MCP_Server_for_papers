# ScholarAgent

ScholarAgent 是一个本地优先的学术研究智能体项目，面向文献检索、论文总结、方法分析、代码复现和学术写作。项目集成了 LangGraph 多智能体工作流、本地论文库检索、arXiv 检索、受控代码/文档工作区、FastAPI Web UI 以及 MCP Server。

项目默认把论文、数据库、向量索引、日志、模型和生成文件保存在本机；除非你显式配置外部 LLM、arXiv 或 MCP 服务，否则运行时数据不会离开本地工作目录。

## 功能特性

- 多智能体工作流：Supervisor、Literature、Summarizer、Methodology、Code Builder、Writing Editor、Database Manager 和 Synthesis。
- 本地论文库：导入 PDF、SQLite 元数据存储、ChromaDB 分块索引、去重、删除和历史记录回填。
- 混合检索：BGE-M3 dense embedding、BM25 sparse retrieval、RRF 融合和 BGE reranker。
- arXiv 集成：通过 arXiv API 搜索和下载论文。
- Code Builder：在受控代码工作区内创建/修改项目文件，运行 Python、pytest、shell 命令和外部 MCP 工具。
- Writing Editor：读取和写入 `.docx`、`.tex`、`.bib`、`.md`、`.txt`，并可调用本地 TeX 工具编译 LaTeX。
- Web UI：多轮对话、会话记忆、Markdown 渲染、实时 Agent 状态、工作区选择器和 Python 解释器选择器。
- MCP Server：把 ScholarAgent 工具暴露给兼容 MCP 的客户端。

## 项目结构

核心工作流：

```text
START -> supervisor -> module_executor -> synthesis -> END
```

主要目录：

```text
scholar_agent/
  agents/             LangGraph 状态、图、服务、调度、执行和整合
  agents/experts/     文献、总结、方法、代码、数据库、写作专家
  tools/              本地论文工具、代码工作区工具、写作工具、注册表
  storage/            SQLite、ChromaDB、论文管理、会话记忆
  plugins/            arXiv、PDF 解析、HyDE
  web/                FastAPI 路由和前端静态文件
  skills/             Skill loader 和 skill 模型
main.py               MCP Server 入口
run_web.py            Web Server 入口
```

## 环境要求

- Python 3.11。
- 一个 OpenAI-compatible 的 LLM API，例如 DeepSeek，通过 `LLM_BASE_URL` 配置。
- 本地 BGE-M3 embedding 模型：默认 `workspace/models/bge-m3`。
- 本地 BGE reranker 模型：默认 `workspace/models/bge-reranker-v2-m3`。
- 推荐使用 CUDA 环境运行 `marker-pdf` 和 BGE 模型；CPU 也可运行但会非常慢。
- 如需编译 LaTeX：本机 `PATH` 中需要有 `latexmk`、`xelatex`、`lualatex` 或 `pdflatex`。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

开发模式：

```powershell
python -m pip install -e ".[dev]"
```

## 本地模型

使用本地论文导入或本地检索前，需要提前下载两个 BGE 模型：

```text
workspace/models/bge-m3
workspace/models/bge-reranker-v2-m3
```

本地论文全部一键导入可以单独运行`scholar_agent/storage/paper_manager.py`。

`bge-m3` 是必需模型：PDF 导入时会把论文分块写入向量库，需要 embedding。`bge-reranker-v2-m3` 用于本地搜索重排；如果缺失，搜索会退回到未重排结果。

如果模型放在其他位置，在 `.env` 中配置：

```text
BGE_M3_MODEL_PATH=D:/models/bge-m3
BGE_RERANKER_MODEL_PATH=D:/models/bge-reranker-v2-m3
PAPER_PARSER_DEVICE=auto
```

`PAPER_PARSER_DEVICE=auto` 会优先使用 CUDA，CUDA 不可用时回退到 CPU。只有当前 Python 环境安装了 CUDA 版 PyTorch 时，才建议显式设置为 `cuda`。

## 配置

复制 `.env.example` 为 `.env` 并修改本地值：

```powershell
Copy-Item .env.example .env
```

最小 LLM 配置：

```text
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat
```

常用路径配置：

```text
WORKSPACE_DIR=./workspace
PAPERS_DIR=./workspace/papers
BGE_M3_MODEL_PATH=./workspace/models/bge-m3
BGE_RERANKER_MODEL_PATH=./workspace/models/bge-reranker-v2-m3
SCHOLAR_AGENT_WORK_ROOT=D:/scholar agent
CODE_BUILDER_WORKSPACE_DIR=D:/scholar agent/scholar code
CODE_BUILDER_PYTHON_EXECUTABLE=D:/
WRITING_WORKSPACE_DIR=D:/scholar agent/scholar document
```

`PAPERS_DIR` 是本地 PDF 论文目录，默认位于 `WORKSPACE_DIR/papers`。如果论文目录独立于项目工作区，可以单独设置 `PAPERS_DIR`。

Web UI 也支持在每次请求中选择 Code Builder 工作区和 Python 解释器。

## workspace 自动创建

仓库不会提交 `workspace/` 目录。新 clone 项目后不需要手动创建它，程序会按需自动创建：

```text
workspace/db/       SQLite 数据库和 ChromaDB 索引
workspace/logs/     运行日志
workspace/models/   默认模型目录
workspace/papers/   默认 PDF 论文目录
```

自动创建发生在：

- Web/MCP 启动时调用 `conf.check_config()`。
- 日志系统启动时自动创建日志目录。
- `PaperDB` 初始化时自动创建数据库目录。
- `VectorDB` 初始化时自动创建向量库目录。
- 论文导入和 arXiv 下载时自动创建论文目录。

需要提前准备的是本地模型文件和待导入的 PDF 文件。

## 启动 Web UI

```powershell
python run_web.py
```

默认地址：

```text
http://127.0.0.1:8000
```

开发热重载：

```text
WEB_RELOAD=true
WEB_RELOAD_DIRS=.
```

## 启动 MCP Server

列出工具：

```powershell
python main.py --mode list-tools
```

以 stdio 启动 MCP Server：

```powershell
python main.py --mode server --transport stdio
```

外部 MCP 工具可在 `mcp_servers.yaml` 或 `mcp_servers.json` 中配置。只做本地 smoke test 时，可关闭外部 MCP 发现：

```powershell
$env:ENABLE_EXTERNAL_MCP='false'
```

MATLAB MCP 的 Python 解释器建议在 `.env` 中配置：

```text
MATLAB_MCP_SERVER_PATH=path/to/matlab_mcp_server.py
```

## Web API

常用接口：


| 方法     | 路径                            | 用途                 |
| -------- | ------------------------------- | -------------------- |
| `GET`    | `/api/health`                   | 健康检查             |
| `GET`    | `/api/tools`                    | 列出已注册工具       |
| `GET`    | `/api/tools/schemas`            | 列出工具 schema      |
| `POST`   | `/api/chat`                     | 阻塞式多智能体对话   |
| `POST`   | `/api/chat/stream`              | SSE 流式多智能体对话 |
| `GET`    | `/api/chat/memory/{session_id}` | 查看会话记忆         |
| `DELETE` | `/api/chat/memory/{session_id}` | 清除会话记忆         |
| `GET`    | `/api/filesystem/roots`         | 获取目录选择器根路径 |
| `GET`    | `/api/filesystem/directories`   | 浏览目录             |
| `GET`    | `/api/filesystem/python-files`  | 浏览 Python 解释器   |
| `GET`    | `/api/local/database`           | 列出本地论文库       |
| `POST`   | `/api/local/database/search`    | 搜索本地论文元数据   |
| `POST`   | `/api/local/search`             | 搜索本地论文全文分块 |
| `POST`   | `/api/papers/import`            | 导入本地 PDF         |
| `POST`   | `/api/papers/delete`            | 删除本地论文         |
| `POST`   | `/api/papers/dedup`             | 检测或清理重复论文   |
| `POST`   | `/api/papers/backfill`          | 回填去重元数据       |
| `POST`   | `/api/arxiv/search`             | 搜索 arXiv           |
| `POST`   | `/api/arxiv/download`           | 下载 arXiv PDF       |

## Skills

Skills 由仓库根目录的 `skill.json` 注册。当前内置示例使用：

```text
source: ./skills
skills/example/SKILL.md
```

新增 skill 时，创建 `skills/<skill-name>/SKILL.md`，并把 `<skill-name>` 加到根目录 `skill.json` 的 skills 列表中。路径应保持相对仓库根目录。

## 测试

编译检查：

```powershell
python -m compileall -q scholar_agent tests main.py run_web.py
```

运行无真实 LLM/网络依赖的工作流检查：

```powershell
python tests/run_agent_workflow_checks.py
```

运行 pytest：

```powershell
python -m pytest tests
```

检查前端 JavaScript 语法：

```powershell
node --check scholar_agent/web/static/app.js
```

## License

MIT
