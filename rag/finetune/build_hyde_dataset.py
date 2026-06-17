"""Build a JSONL dataset for fine-tuning a local HyDE generator.

Each output row has:
  {"query": "...", "passage": "...", "paper_id": "...", "title": "...", "source": "..."}

The script uses local paper metadata plus existing vector chunks. It does not
call an LLM; it creates deterministic weak-supervision pairs such as
title -> chunk and title + tags -> chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from config import conf
from rag.storage import PaperManager


def _clean_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:max_chars].strip()


def _queries_for_paper(paper: dict) -> Iterable[tuple[str, str]]:
    title = _clean_text(paper.get("title", ""), 220)
    authors = _clean_text(paper.get("authors", ""), 160)
    tags = _clean_text(paper.get("tags", ""), 160)
    if title:
        yield title, "title"
        yield f"What is the main method and contribution of {title}?", "title_question"
    if title and tags:
        yield f"{title} {tags}", "title_tags"
    if title and authors:
        yield f"{title} by {authors}", "title_authors"


def build_dataset(output: Path, max_chunks_per_paper: int, max_passage_chars: int) -> int:
    manager = PaperManager(enable_hyde=False)
    papers = manager.list_all()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[tuple[str, str]] = set()

    with output.open("w", encoding="utf-8") as fh:
        for paper in papers:
            paper_id = str(paper.get("id") or "").strip()
            if not paper_id:
                continue
            chunks = manager.get_chunks_for_paper_ids(
                [paper_id],
                max_chunks_per_paper=max_chunks_per_paper,
            )
            for chunk in chunks:
                passage = _clean_text(chunk.get("content", ""), max_passage_chars)
                if len(passage) < 80:
                    continue
                for query, source in _queries_for_paper(paper):
                    query = _clean_text(query, 300)
                    key = (query, passage[:120])
                    if not query or key in seen:
                        continue
                    seen.add(key)
                    row = {
                        "query": query,
                        "passage": passage,
                        "paper_id": paper_id,
                        "title": paper.get("title", ""),
                        "source": source,
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local HyDE fine-tuning JSONL dataset.")
    default_output = Path(conf.PROJECT_ROOT) / "rag" / "finetune" / "data" / "hyde_train.jsonl"
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--max-chunks-per-paper", type=int, default=8)
    parser.add_argument("--max-passage-chars", type=int, default=900)
    args = parser.parse_args()

    count = build_dataset(
        output=Path(args.output),
        max_chunks_per_paper=max(1, args.max_chunks_per_paper),
        max_passage_chars=max(120, args.max_passage_chars),
    )
    print(f"Wrote {count} HyDE training rows to {args.output}")


if __name__ == "__main__":
    main()
