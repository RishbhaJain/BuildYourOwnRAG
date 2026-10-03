import json
import math

import pytest

from retrieval_evaluation import RetrievalCase, evaluate, load_cases, main


def test_evaluate_reports_standard_ranking_metrics():
    cases = [
        RetrievalCase("q1", ("b", "d"), ("a", "b", "c", "d")),
        RetrievalCase("q2", ("x",), ("x", "y")),
    ]

    report = evaluate(cases, [1, 4])

    assert report["query_count"] == 2
    assert report["metrics"]["hit_rate@1"] == pytest.approx(0.5)
    assert report["metrics"]["recall@1"] == pytest.approx(0.5)
    assert report["metrics"]["mrr@4"] == pytest.approx(0.75)
    expected_q1_ndcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal_q1_ndcg = 1 + 1 / math.log2(3)
    assert report["per_query"][0]["metrics"]["ndcg@4"] == pytest.approx(
        expected_q1_ndcg / ideal_q1_ndcg
    )
    assert report["per_query"][1]["metrics"]["ndcg@4"] == 1.0


def test_evaluate_counts_no_hit_as_zero():
    report = evaluate([RetrievalCase("q", ("relevant",), ("other",))], [1])
    assert report["metrics"] == {
        "hit_rate@1": 0.0,
        "recall@1": 0.0,
        "mrr@1": 0.0,
        "ndcg@1": 0.0,
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"query_id": "q", "relevant_chunk_ids": [], "retrieved_chunk_ids": []},
            "must not be empty",
        ),
        (
            {
                "query_id": "q",
                "relevant_chunk_ids": ["a", "a"],
                "retrieved_chunk_ids": [],
            },
            "duplicates",
        ),
        (
            {
                "query_id": "q",
                "relevant_chunk_ids": ["a"],
                "retrieved_chunk_ids": ["b", "b"],
            },
            "duplicates",
        ),
    ],
)
def test_load_cases_rejects_invalid_labels_and_rankings(tmp_path, payload, message):
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_cases(run)


def test_load_cases_rejects_duplicate_query_ids(tmp_path):
    run = tmp_path / "run.jsonl"
    record = {
        "query_id": "q",
        "relevant_chunk_ids": ["a"],
        "retrieved_chunk_ids": ["a"],
    }
    run.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_cases(run)


def test_cli_writes_artifact_and_enforces_regression_threshold(tmp_path):
    run = tmp_path / "run.jsonl"
    output = tmp_path / "report.json"
    run.write_text(
        json.dumps(
            {
                "query_id": "q",
                "relevant_chunk_ids": ["a"],
                "retrieved_chunk_ids": ["b", "a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [str(run), "--ks", "1,2", "--output", str(output), "--minimum", "recall@2=1"]
    )
    assert exit_code == 0
    assert json.loads(output.read_text())["metrics"]["mrr@2"] == 0.5

    assert main([str(run), "--ks", "1", "--minimum", "recall@1=0.5"]) == 1


def test_unknown_threshold_metric_is_rejected(tmp_path):
    run = tmp_path / "run.jsonl"
    run.write_text(
        json.dumps(
            {"query_id": "q", "relevant_chunk_ids": ["a"], "retrieved_chunk_ids": ["a"]}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown metric"):
        main([str(run), "--ks", "1", "--minimum", "precision@1=0.5"])
