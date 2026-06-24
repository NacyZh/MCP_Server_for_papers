# HyDE Fine-Tuning

[English](README.md) | [简体中文](README_zh.md)

本目录包含用于微调本地 HyDE generator 的工具。该模型用于 RAG 检索流程。

HyDE 会根据查询生成一段紧凑的假想论文片段。生成片段会追加到原始查询中用于 dense retrieval，而 BM25 和 reranking 仍可使用原始查询。

## 目录结构

```text
rag/finetune/
  data/                         生成的训练数据，Git 忽略
  models/                       LoRA adapter 或 merged model，Git 忽略
  build_hyde_dataset.py         从本地论文构建弱监督 JSONL
  train_hyde_lora.py            准备/运行 LLaMA-Factory LoRA 训练
  evaluate_hyde_recall.py       对比 HyDE 开启/关闭时的 retrieval recall
```

## 工作流

1. 使用 `add_paper_to_database` 导入并索引论文。
2. 从本地论文 chunks 构建弱监督训练样本。
3. 如果重视质量，人工检查并清洗生成的 JSONL。
4. 使用 LLaMA-Factory 训练 LoRA adapter。
5. 用 vLLM 提供训练后的模型服务。
6. 启用 `HYDE_BACKEND=vllm` 并评估检索质量。

## 构建训练数据

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.build_hyde_dataset --output rag/finetune/data/hyde_train.jsonl --max-chunks-per-paper 8 --max-passage-chars 900
```

输出行使用以下 schema：

```json
{"query":"...","passage":"...","paper_id":"local_xxx","title":"...","source":"title_question"}
```

构建器是确定性的，不会调用 LLM。它会从标题、作者、标签和已有 chunks 中创建弱标签。若需要 HyDE，应人工检查并过滤生成样本。

## 使用 LLaMA-Factory 训练

在训练环境中安装可选训练依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install ".[hyde-train]"
```

准备 LLaMA-Factory 数据和配置：

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.train_hyde_lora --model Qwen/Qwen2.5-1.5B-Instruct --train-jsonl rag/finetune/data/hyde_train.jsonl --output-dir rag/finetune/models/hyde-qwen-lora --dataset-dir rag/finetune/data/llamafactory_data --template qwen
```

添加 `--run` 可直接运行训练：

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.train_hyde_lora --model Qwen/Qwen2.5-1.5B-Instruct --train-jsonl rag/finetune/data/hyde_train.jsonl --output-dir rag/finetune/models/hyde-qwen-lora --dataset-dir rag/finetune/data/llamafactory_data --template qwen --bf16 --run
```

根据可用 GPU 调整 `--bf16`、`--fp16`、batch size、gradient accumulation 和 LoRA rank。

## 使用 vLLM 本地推理

在运行环境中安装 vLLM：

```powershell
.\.venv\Scripts\python.exe -m pip install ".[hyde-vllm]"
```

配置 `.env`：

```text
ENABLE_HYDE=true
HYDE_BACKEND=vllm
HYDE_VLLM_MODEL_PATH=./rag/finetune/models/hyde-qwen-lora
HYDE_VLLM_DTYPE=auto
HYDE_VLLM_TENSOR_PARALLEL_SIZE=1
HYDE_VLLM_GPU_MEMORY_UTILIZATION=0.85
HYDE_VLLM_MAX_MODEL_LEN=2048
```

## 评估 HyDE

快速检查 HyDE 开启/关闭对 recall 的影响时，准备如下 JSONL 文件：

```json
{"query":"query","paper_id":"local_xxx"}
```

然后运行：

```powershell
.\.venv\Scripts\python.exe -m rag.finetune.evaluate_hyde_recall --eval-jsonl rag/finetune/data/hyde_eval.jsonl --k 5
```

 retrieval evaluation 推荐使用 `rag/evaluation` 中更完整的评估流程，它会比较 hybrid、dense 和 BM25 模式并写入完整报告。
