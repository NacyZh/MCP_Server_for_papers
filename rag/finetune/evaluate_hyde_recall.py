"""Evaluate local retrieval Recall@K with HyDE on/off.

Input JSONL rows must contain:
  {"query": "...", "paper_id": "local_xxx"}

The script compares retrieval with HyDE disabled and enabled. Configure local
HyDE through .env, for example:
  HYDE_BACKEND=vllm
  HYDE_VLLM_MODEL_PATH=./rag/finetune/models/hyde-qwen-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.storage import PaperManager


def _load_eval_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row.get("query") or "").strip()
            paper_id = str(row.get("paper_id") or "").strip()
            if query and paper_id:
                rows.append({"query": query, "paper_id": paper_id})
    return rows


def _recall_at_k(manager: PaperManager, rows: list[dict], k: int, use_hyde: bool) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        results = manager.search_knowledge(row["query"], n_results=k, use_hyde=use_hyde)
        paper_ids = {str(item.get("paper_id") or "") for item in results}
        if row["paper_id"] in paper_ids:
            hits += 1
    return hits / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HyDE retrieval Recall@K.")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    rows = _load_eval_rows(Path(args.eval_jsonl))
    manager = PaperManager(enable_hyde=True)
    k = max(1, args.k)
    no_hyde = _recall_at_k(manager, rows, k=k, use_hyde=False)
    with_hyde = _recall_at_k(manager, rows, k=k, use_hyde=True)

    print(json.dumps({
        "rows": len(rows),
        "k": k,
        "recall_no_hyde": no_hyde,
        "recall_with_hyde": with_hyde,
        "delta": with_hyde - no_hyde,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
