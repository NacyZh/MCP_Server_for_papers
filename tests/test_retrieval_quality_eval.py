import json

from rag.evaluation import compare_reports, evaluate_retrieval_cases, load_retrieval_cases_jsonl
from rag.evaluation.retrieval_quality import RetrievalCase
from rag.evaluation.run_retrieval_eval import main as run_retrieval_eval_main


def test_evaluate_retrieval_cases_reports_recall_precision_and_mrr():
    cases = [
        RetrievalCase(
            case_id="c1",
            query="attention",
            expected_paper_ids=["local_a"],
            expected_chunk_ids=["local_a_chunk_2"],
        ),
        RetrievalCase(
            case_id="c2",
            query="moe",
            expected_sections=["Method"],
        ),
    ]

    def search(query, k):
        if query == "attention":
            return [
                {"paper_id": "local_x", "chunk_id": "local_x_chunk_1", "section_title": "Intro"},
                {"paper_id": "local_a", "chunk_id": "local_a_chunk_2", "section_title": "Method"},
            ][:k]
        return [
            {"paper_id": "local_b", "chunk_id": "local_b_chunk_1", "section_title": "Method"},
        ][:k]

    report = evaluate_retrieval_cases(cases, search, k_values=(1, 2))

    assert report["case_count"] == 2
    assert report["aggregate"]["paper_recall@1"] == 0.0
    assert report["aggregate"]["paper_recall@2"] == 0.5
    assert report["aggregate"]["chunk_recall@2"] == 0.5
    assert report["aggregate"]["section_hit@1"] == 0.5
    assert report["aggregate"]["mrr"] == 0.75


def test_compare_reports_returns_metric_deltas():
    baseline = {"case_count": 1, "aggregate": {"mrr": 0.5, "paper_recall@5": 1.0}}
    candidate = {"case_count": 1, "aggregate": {"mrr": 0.75, "paper_recall@5": 0.5}}

    diff = compare_reports(baseline, candidate)

    assert diff["deltas"]["mrr"]["delta"] == 0.25
    assert diff["deltas"]["paper_recall@5"]["delta"] == -0.5
    assert "paper_recall@5" in diff["regressions"]
    assert "mrr" in diff["improvements"]


def test_load_retrieval_cases_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [
        {"query": "attention", "expected_paper_ids": ["local_a"]},
        {"id": "custom", "query": "moe", "expected_sections": ["Method"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cases = load_retrieval_cases_jsonl(path)

    assert len(cases) == 2
    assert cases[0].case_id == "case_1"
    assert cases[1].case_id == "custom"
    assert cases[1].expected_sections == ["Method"]


def test_load_retrieval_cases_jsonl_requires_expectations(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"query": "attention"}), encoding="utf-8")

    try:
        load_retrieval_cases_jsonl(path)
    except ValueError as exc:
        assert "expected_*" in str(exc)
    else:
        raise AssertionError("missing expectations should fail")


def test_retrieval_eval_cli_requires_real_dataset(tmp_path, capsys):
    exit_code = run_retrieval_eval_main([
        "--dataset",
        "rag/evaluation/data/missing.jsonl",
        "--results-dir",
        str(tmp_path),
    ])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Evaluation dataset not found" in captured.err
