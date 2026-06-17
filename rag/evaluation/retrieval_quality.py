"""Small, deterministic retrieval-quality evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_paper_ids: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sections: List[str] = field(default_factory=list)
    case_id: str = ""

    @property
    def has_expectations(self) -> bool:
        return bool(self.expected_paper_ids or self.expected_chunk_ids or self.expected_sections)


def load_retrieval_cases_jsonl(path: str | Path, *, max_cases: int | None = None) -> List[RetrievalCase]:
    cases: List[RetrievalCase] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSONL row: {exc}") from exc
            query = str(row.get("query") or "").strip()
            if not query:
                raise ValueError(f"{source}:{line_no}: query is required")
            case = RetrievalCase(
                case_id=str(row.get("case_id") or row.get("id") or f"case_{line_no}"),
                query=query,
                expected_paper_ids=_as_str_list(row.get("expected_paper_ids")),
                expected_chunk_ids=_as_str_list(row.get("expected_chunk_ids")),
                expected_sections=_as_str_list(row.get("expected_sections")),
            )
            if not case.has_expectations:
                raise ValueError(f"{source}:{line_no}: at least one expected_* field is required")
            cases.append(case)
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases


def evaluate_retrieval_cases(
    cases: Sequence[RetrievalCase],
    search_fn: Callable[[str, int], Sequence[Dict[str, Any]]],
    *,
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    normalized_k = sorted({max(1, int(k)) for k in k_values})
    if not normalized_k:
        normalized_k = [5]
    max_k = max(normalized_k)
    totals = {
        f"paper_recall@{k}": 0.0 for k in normalized_k
    } | {
        f"chunk_recall@{k}": 0.0 for k in normalized_k
    } | {
        f"section_hit@{k}": 0.0 for k in normalized_k
    } | {
        f"precision@{k}": 0.0 for k in normalized_k
    }
    reciprocal_rank_sum = 0.0
    per_case: List[Dict[str, Any]] = []

    for case in cases:
        results = [dict(item) for item in search_fn(case.query, max_k)]
        summary = _evaluate_one(case, results, normalized_k)
        reciprocal_rank_sum += float(summary["mrr"])
        for key, value in summary["metrics"].items():
            totals[key] += float(value)
        per_case.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "metrics": summary["metrics"],
                "mrr": summary["mrr"],
                "top_results": [
                    {
                        "rank": idx + 1,
                        "paper_id": item.get("paper_id"),
                        "chunk_id": item.get("chunk_id"),
                        "section_name": item.get("section_name"),
                        "section_title": item.get("section_title"),
                        "score": item.get("score", item.get("rerank_score", item.get("rrf_score"))),
                    }
                    for idx, item in enumerate(results[:max_k])
                ],
            }
        )

    count = len(cases)
    aggregate = {key: (value / count if count else 0.0) for key, value in totals.items()}
    aggregate["mrr"] = reciprocal_rank_sum / count if count else 0.0
    return {
        "case_count": count,
        "k_values": normalized_k,
        "aggregate": aggregate,
        "cases": per_case,
    }


def compare_reports(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    base_metrics = baseline.get("aggregate") or {}
    cand_metrics = candidate.get("aggregate") or {}
    metric_names = sorted(set(base_metrics) | set(cand_metrics))
    deltas = {
        name: {
            "baseline": float(base_metrics.get(name, 0.0)),
            "candidate": float(cand_metrics.get(name, 0.0)),
            "delta": float(cand_metrics.get(name, 0.0)) - float(base_metrics.get(name, 0.0)),
        }
        for name in metric_names
    }
    regressions = {
        name: values
        for name, values in deltas.items()
        if values["delta"] < 0
    }
    improvements = {
        name: values
        for name, values in deltas.items()
        if values["delta"] > 0
    }
    return {
        "baseline_case_count": baseline.get("case_count", 0),
        "candidate_case_count": candidate.get("case_count", 0),
        "deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
    }


def _evaluate_one(case: RetrievalCase, results: Sequence[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, Any]:
    expected_papers = set(case.expected_paper_ids)
    expected_chunks = set(case.expected_chunk_ids)
    expected_sections = {s.lower() for s in case.expected_sections}
    metrics: Dict[str, float] = {}

    relevant_ranks: List[int] = []
    for rank, item in enumerate(results, start=1):
        if _is_relevant(item, expected_papers, expected_chunks, expected_sections):
            relevant_ranks.append(rank)

    for k in k_values:
        top = list(results[:k])
        papers = {str(item.get("paper_id") or "") for item in top}
        chunks = {str(item.get("chunk_id") or "") for item in top}
        sections = {
            str(item.get("section_name") or "").lower()
            for item in top
        } | {
            str(item.get("section_title") or "").lower()
            for item in top
        }
        relevant_at_k = sum(
            1
            for item in top
            if _is_relevant(item, expected_papers, expected_chunks, expected_sections)
        )
        metrics[f"paper_recall@{k}"] = _recall(expected_papers, papers)
        metrics[f"chunk_recall@{k}"] = _recall(expected_chunks, chunks)
        metrics[f"section_hit@{k}"] = 1.0 if expected_sections and expected_sections & sections else 0.0
        metrics[f"precision@{k}"] = relevant_at_k / k if k else 0.0

    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    return {"metrics": metrics, "mrr": mrr}


def _is_relevant(
    item: Dict[str, Any],
    expected_papers: set[str],
    expected_chunks: set[str],
    expected_sections: set[str],
) -> bool:
    paper_hit = bool(expected_papers and str(item.get("paper_id") or "") in expected_papers)
    chunk_hit = bool(expected_chunks and str(item.get("chunk_id") or "") in expected_chunks)
    section_values = {
        str(item.get("section_name") or "").lower(),
        str(item.get("section_title") or "").lower(),
    }
    section_hit = bool(expected_sections and expected_sections & section_values)
    return paper_hit or chunk_hit or section_hit


def _recall(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & observed) / len(expected)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if str(item or "").strip()]
