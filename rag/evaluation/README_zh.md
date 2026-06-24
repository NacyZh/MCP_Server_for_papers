# Retrieval Evaluation

[English](README.md) | [简体中文](README_zh.md)

本目录包含 RAG MCP server 的本地检索质量评估流程。

## 目录结构

```text
rag/evaluation/
  data/
    retrieval_eval.sample.jsonl   已提交的样例 schema
    retrieval_eval.jsonl          真实评估集
  results/                        JSON 报告
  retrieval_quality.py            指标辅助函数
  run_retrieval_eval.py           命令行评估脚本
```

## 数据集

真实评估数据应手动收集和标注。样例文件只作为 schema 参考，不应作为质量 benchmark。

每个 JSONL 行必须包含 `query`，并至少包含一个 expected 字段：

```json
{"case_id":"example_001","query":"example query","expected_paper_ids":["local_xxx"],"expected_sections":["example section"]}
```

支持的 expected 字段：

- `expected_paper_ids`：由 `list_local_database` 或 `search_local_database` 返回的论文 ID。
- `expected_chunk_ids`：可用时填写精确 chunk ID。
- `expected_sections`：期望被检索到的章节名或章节标题。

建议先准备 20-50 条高质量 case，覆盖真实使用场景：

- 标题和元数据查找
- 方法和系统模型查找
- 公式和算法查找
- 实验和结果查找
- 跨论文比较

## 运行评估

从样例创建真实数据集：

```powershell
Copy-Item rag/evaluation/data/retrieval_eval.sample.jsonl rag/evaluation/data/retrieval_eval.jsonl
```

将 `local_replace_me` 替换为真实本地 ID，然后运行：

```powershell
.\.venv\Scripts\python.exe -m rag.evaluation.run_retrieval_eval --dataset rag/evaluation/data/retrieval_eval.jsonl --modes hybrid,dense,bm25 --top-k 10 --max-cases 200
```

脚本会将完整报告写入：

```text
rag/evaluation/results/retrieval_eval_<timestamp>.json
```

## 指标

报告包含：

- `paper_recall@k`
- `chunk_recall@k`
- `section_hit@k`
- `precision@k`
- `mrr`
- 不同检索模式之间的指标差异

除非明确测试 dense-only 或 BM25-only 行为，否则建议使用 `hybrid` 作为默认 baseline。
