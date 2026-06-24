# Retrieval Evaluation

[English](README.md) | [简体中文](README_zh.md)

This directory contains the local retrieval-quality evaluation workflow for the RAG MCP server.

## Directory Layout

```text
rag/evaluation/
  data/
    retrieval_eval.sample.jsonl   sample schema, committed
    retrieval_eval.jsonl          real evaluation set
  results/                        JSON reports
  retrieval_quality.py            metric helpers
  run_retrieval_eval.py           command-line evaluation script
```

## Dataset

Real evaluation data should be manually collected and labeled. The sample file is only a schema reference and should not be used as a quality benchmark.

Each JSONL row must contain `query` and at least one expected field:

```json
{"case_id":"example_001","query":"example query","expected_paper_ids":["local_xxx"],"expected_sections":["example section"]}
```

Supported expected fields:

- `expected_paper_ids`: paper IDs returned by `list_local_database` or `search_local_database`.
- `expected_chunk_ids`: exact chunk IDs when available.
- `expected_sections`: section names or titles that should be retrieved.

Start with 20-50 high-quality cases that reflect actual usage:

- title and metadata lookup
- method and system-model lookup
- formula and algorithm lookup
- experiment and result lookup
- cross-paper comparison

## Run Evaluation

Create the real dataset from the sample:

```powershell
Copy-Item rag/evaluation/data/retrieval_eval.sample.jsonl rag/evaluation/data/retrieval_eval.jsonl
```

Replace `local_replace_me` values with real local IDs, then run:

```powershell
.\.venv\Scripts\python.exe -m rag.evaluation.run_retrieval_eval --dataset rag/evaluation/data/retrieval_eval.jsonl --modes hybrid,dense,bm25 --top-k 10 --max-cases 200
```

The script writes a full report to:

```text
rag/evaluation/results/retrieval_eval_<timestamp>.json
```

## Metrics

The report includes:

- `paper_recall@k`
- `chunk_recall@k`
- `section_hit@k`
- `precision@k`
- `mrr`
- metric deltas between retrieval modes

Use `hybrid` as the default baseline unless you are explicitly testing dense-only or BM25-only behavior.
