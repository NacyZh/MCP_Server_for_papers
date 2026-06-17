"""Run retrieval-quality evaluation against the local RAG database.

Example:
    python -m rag.evaluation.run_retrieval_eval \
        --dataset rag/evaluation/data/retrieval_eval.jsonl \
        --modes hybrid,dense,bm25 \
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import conf
from rag.core.logging import configure_logging, get_logger
from rag.evaluation import compare_reports, evaluate_retrieval_cases, load_retrieval_cases_jsonl
from rag.tools.security import ToolSecurityError, resolve_project_file

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval quality.")
    parser.add_argument(
        "--dataset",
        default=conf.RETRIEVAL_EVAL_DATASET_PATH,
        help="Project-relative JSONL dataset path.",
    )
    parser.add_argument(
        "--results-dir",
        default=conf.RETRIEVAL_EVAL_RESULTS_DIR,
        help="Directory where JSON reports are written.",
    )
    parser.add_argument(
        "--modes",
        default="hybrid,dense,bm25",
        help="Comma-separated retrieval modes: hybrid,dense,bm25.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Largest k for recall/precision metrics.")
    parser.add_argument("--max-cases", type=int, default=conf.RETRIEVAL_EVAL_MAX_CASES)
    parser.add_argument("--use-hyde", action="store_true", help="Enable HyDE expansion during evaluation.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit output JSON path. Defaults to results-dir/retrieval_eval_<timestamp>.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    try:
        dataset = resolve_project_file(args.dataset, conf.PROJECT_ROOT, suffix=".jsonl")
    except ToolSecurityError as exc:
        print(f"Invalid dataset path: {exc}", file=sys.stderr)
        return 2
    if not dataset.is_file():
        sample = Path(conf.PROJECT_ROOT) / "rag" / "evaluation" / "data" / "retrieval_eval.sample.jsonl"
        print(
            f"Evaluation dataset not found: {dataset}\n"
            f"Create it from the sample file and replace expected IDs with real local IDs: {sample}",
            file=sys.stderr,
        )
        return 2

    modes = _parse_modes(args.modes)
    if not modes:
        print("No retrieval modes selected.", file=sys.stderr)
        return 2

    top_k = max(1, min(int(args.top_k), 50))
    max_cases = max(1, min(int(args.max_cases), conf.RETRIEVAL_EVAL_MAX_CASES))
    cases = load_retrieval_cases_jsonl(dataset, max_cases=max_cases)
    if not cases:
        print(f"Evaluation dataset is empty: {dataset}", file=sys.stderr)
        return 2

    from rag.storage.paper_manager import PaperManager

    pm = PaperManager(enable_hyde=bool(args.use_hyde))
    k_values = sorted({k for k in (1, 3, 5, 10, top_k) if k <= top_k})
    reports: dict[str, dict[str, Any]] = {}
    for mode in modes:
        logger.info("[eval] running mode=%s cases=%s top_k=%s", mode, len(cases), top_k)

        def search_fn(query: str, k: int, mode: str = mode) -> list[dict[str, Any]]:
            return pm.search_knowledge(
                query,
                n_results=k,
                use_hyde=bool(args.use_hyde),
                retrieval_mode=mode,
                rerank=not bool(args.no_rerank),
            )

        reports[mode] = evaluate_retrieval_cases(cases, search_fn, k_values=k_values)

    baseline = modes[0]
    diffs = {
        mode: compare_reports(reports[baseline], report)
        for mode, report in reports.items()
        if mode != baseline
    }
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(dataset),
        "modes": modes,
        "baseline_mode": baseline,
        "top_k": top_k,
        "max_cases": max_cases,
        "use_hyde": bool(args.use_hyde),
        "rerank": not bool(args.no_rerank),
        "reports": reports,
        "diffs": diffs,
    }
    output = _resolve_output_path(args.output, args.results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(_format_summary(payload))
    print(f"Report written: {output}")
    return 0


def _parse_modes(raw: str) -> list[str]:
    modes = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in {"hybrid", "dense", "bm25"}]
    if invalid:
        raise SystemExit(f"Unsupported retrieval mode(s): {', '.join(invalid)}")
    return modes


def _resolve_output_path(output: str, results_dir: str) -> Path:
    if output:
        path = Path(output).expanduser()
        if not path.is_absolute():
            path = Path(conf.PROJECT_ROOT) / path
        return path.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (Path(results_dir).expanduser().resolve() / f"retrieval_eval_{timestamp}.json")


def _format_summary(payload: dict[str, Any]) -> str:
    lines = [
        "=== Retrieval Evaluation Summary ===",
        f"Dataset: {payload['dataset']}",
        f"Cases: {next(iter(payload['reports'].values()))['case_count'] if payload['reports'] else 0}",
        f"Baseline: {payload['baseline_mode']}",
    ]
    top_k = payload["top_k"]
    for mode, report in payload["reports"].items():
        aggregate = report.get("aggregate") or {}
        lines.append(
            f"- {mode}: mrr={aggregate.get('mrr', 0.0):.4f}, "
            f"paper_recall@{top_k}={aggregate.get(f'paper_recall@{top_k}', 0.0):.4f}, "
            f"chunk_recall@{top_k}={aggregate.get(f'chunk_recall@{top_k}', 0.0):.4f}, "
            f"precision@{top_k}={aggregate.get(f'precision@{top_k}', 0.0):.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
