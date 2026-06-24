# HyDE Fine-Tuning

[English](README.md) | [简体中文](README_zh.md)

This directory contains utilities for fine-tuning a local HyDE generator used by the RAG retrieval pipeline.

HyDE generates a compact hypothetical paper passage from a query. The generated passage is appended to the original query for dense retrieval, while BM25 and reranking can still use the original query.

## Directory Layout

```text
rag/finetune/
  data/                         generated training data, ignored by Git
  models/                       LoRA adapters or merged models, ignored by Git
  build_hyde_dataset.py         build weak-supervision JSONL from local papers
  train_hyde_lora.py            prepare/run LLaMA-Factory LoRA training
  evaluate_hyde_recall.py       compare retrieval recall with HyDE on/off
```

## Workflow

1. Import and index papers with `add_paper_to_database`.
2. Build weak-supervision training rows from local paper chunks.
3. Review and clean the generated JSONL if quality matters.
4. Train a LoRA adapter with LLaMA-Factory.
5. Serve the trained model with vLLM.
6. Enable `HYDE_BACKEND=vllm` and evaluate retrieval quality.

## Build Training Data

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.build_hyde_dataset --output rag/finetune/data/hyde_train.jsonl --max-chunks-per-paper 8 --max-passage-chars 900
```

Output rows use this schema:

```json
{"query":"...","passage":"...","paper_id":"local_xxx","title":"...","source":"title_question"}
```

The builder is deterministic and does not call an LLM. It creates weak labels from titles, authors, tags, and existing chunks. For HyDE, manually inspect and filter the generated rows.

## Train With LLaMA-Factory

Install the optional training dependencies in the environment you use for training:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[hyde-train]"
```

Prepare LLaMA-Factory data and config:

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.train_hyde_lora --model Qwen/Qwen2.5-1.5B-Instruct --train-jsonl rag/finetune/data/hyde_train.jsonl --output-dir rag/finetune/models/hyde-qwen-lora --dataset-dir rag/finetune/data/llamafactory_data --template qwen
```

Run training directly by adding `--run`:

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.train_hyde_lora --model Qwen/Qwen2.5-1.5B-Instruct --train-jsonl rag/finetune/data/hyde_train.jsonl --output-dir rag/finetune/models/hyde-qwen-lora --dataset-dir rag/finetune/data/llamafactory_data --template qwen --bf16 --run
```

Adjust `--bf16`, `--fp16`, batch size, gradient accumulation, and LoRA rank for the available GPU.

## Local Inference With vLLM

Install vLLM in the runtime environment:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[hyde-vllm]"
```

Configure `.env`:

```text
ENABLE_HYDE=true
HYDE_BACKEND=vllm
HYDE_VLLM_MODEL_PATH=./rag/finetune/models/hyde-qwen-lora
HYDE_VLLM_DTYPE=auto
HYDE_VLLM_TENSOR_PARALLEL_SIZE=1
HYDE_VLLM_GPU_MEMORY_UTILIZATION=0.85
HYDE_VLLM_MAX_MODEL_LEN=2048
```

## Evaluate HyDE

For a quick HyDE on/off recall check, prepare a JSONL file with:

```json
{"query":"query","paper_id":"local_xxx"}
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.evaluate_hyde_recall --eval-jsonl rag/finetune/data/hyde_eval.jsonl --k 5
```

For production retrieval evaluation, prefer the broader evaluation workflow in `rag/evaluation`, which compares hybrid, dense, and BM25 modes and writes full reports.
