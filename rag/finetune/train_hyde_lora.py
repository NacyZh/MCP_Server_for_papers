"""Prepare and optionally launch HyDE LoRA training with LLaMA-Factory.

Expected input JSONL rows:
  {"query": "...", "passage": "..."}

This script converts rows to LLaMA-Factory alpaca format, writes
dataset_info.json and a train YAML, then optionally runs:
  llamafactory-cli train <yaml>

Example:
  python -m rag.finetune.train_hyde_lora --model Qwen/Qwen2.5-1.5B-Instruct \
      --train-jsonl rag/finetune/data/hyde_train.jsonl \
      --output-dir rag/finetune/models/hyde-qwen-lora \
      --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

SYSTEM_PROMPT = (
    "You are a scholarly retrieval assistant. Given a research question or "
    "keyword phrase, write a single compact passage that could plausibly "
    "appear in a peer-reviewed paper directly addressing the question. "
    "Do not add headings, bullets, citations, markdown, or explanations."
)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row.get("query") or "").strip()
            passage = str(row.get("passage") or "").strip()
            if query and passage:
                rows.append({"query": query, "passage": passage})
    return rows


def _write_alpaca_dataset(rows: list[dict], dataset_dir: Path, dataset_name: str) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / f"{dataset_name}.json"
    records = []
    for row in rows:
        records.append(
            {
                "instruction": (
                    "Generate one compact paper-like hypothetical passage for dense retrieval. "
                    "Do not add headings, bullets, citations, markdown, or explanations."
                ),
                "input": f"Query: {row['query']}\n\nPassage:",
                "output": row["passage"],
                "system": SYSTEM_PROMPT,
            }
        )
    dataset_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    info_path = dataset_dir / "dataset_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info = {}
    else:
        info = {}
    info[dataset_name] = {
        "file_name": dataset_path.name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_path


def _write_train_yaml(args, dataset_dir: Path, dataset_name: str) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "llamafactory_hyde_lora.yaml"
    config = {
        "model_name_or_path": args.model,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "lora_target": args.lora_target,
        "dataset_dir": str(dataset_dir),
        "dataset": dataset_name,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "max_samples": args.max_samples,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "output_dir": str(output_dir),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": "none",
    }
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/run HyDE LoRA training with LLaMA-Factory.")
    parser.add_argument("--model", required=True, help="Base HF model or local path.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default="rag/finetune/data/llamafactory_data")
    parser.add_argument("--dataset-name", default="hyde_train")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-target", default="all")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run llamafactory-cli train after writing files.")
    args = parser.parse_args()

    rows = _load_rows(Path(args.train_jsonl))
    if not rows:
        raise SystemExit(f"No usable rows found in {args.train_jsonl}")

    dataset_dir = Path(args.dataset_dir)
    dataset_path = _write_alpaca_dataset(rows, dataset_dir=dataset_dir, dataset_name=args.dataset_name)
    yaml_path = _write_train_yaml(args, dataset_dir=dataset_dir, dataset_name=args.dataset_name)

    print(f"Wrote LLaMA-Factory dataset: {dataset_path}")
    print(f"Wrote LLaMA-Factory train config: {yaml_path}")
    print(f"Train command: llamafactory-cli train {yaml_path}")
    if args.run:
        subprocess.run(["llamafactory-cli", "train", str(yaml_path)], check=True)


if __name__ == "__main__":
    main()
