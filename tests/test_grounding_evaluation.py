import json

import pytest

from grounding_evaluation import (
    Context,
    GroundingCase,
    content_tokens,
    evaluate,
    load_cases,
    main,
    tokenize,
)


def test_tokenize_normalizes_unicode_case_and_numbers():
    assert tokenize("Caf\u00e9 COSTS $1,200.50") == ["caf\u00e9", "costs", "1,200.50"]


def test_content_tokens_remove_function_words_without_emptying_answer():
    assert content_tokens("The model is in Berkeley") == ["model", "berkeley"]
    assert content_tokens("The") == ["the"]


def test_evaluate_reports_union_and_single_context_support():
    case = GroundingCase(
        "q1",
        "Dan Garcia leads GamesCrafters",
        (
            Context("a", "Dan Garcia is a professor."),
            Context("b", "He leads the GamesCrafters group."),
        ),
    )

    report = evaluate([case])

    assert report["metrics"]["mean_token_support"] == 1.0
    assert report["metrics"]["fully_supported_answer_rate"] == 1.0
    assert report["per_query"][0]["max_context_support"] == pytest.approx(0.5)
    assert report["per_query"][0]["supporting_chunk_ids"] == ["a", "b"]


def test_evaluate_exposes_unsupported_tokens_and_numbers():
    case = GroundingCase(
        "q1",
        "The lab opened in 2025 in Oakland",
        (Context("a", "The lab opened in 2024 in Berkeley."),),
    )

    report = evaluate([case])
    row = report["per_query"][0]

    assert row["unsupported_tokens"] == ["2025", "oakland"]
    assert row["unsupported_numeric_tokens"] == ["2025"]
    assert report["metrics"]["unsupported_number_rate"] == 1.0


def test_abstentions_are_reported_but_not_rewarded_as_grounded():
    report = evaluate(
        [
            GroundingCase("q1", "Unknown", (Context("a", "No evidence."),)),
            GroundingCase("q2", "Berkeley", (Context("b", "UC Berkeley"),)),
        ]
    )

    assert report["answered_count"] == 1
    assert report["abstained_count"] == 1
    assert report["metrics"]["abstention_rate"] == 0.5
    assert report["metrics"]["mean_token_support"] == 1.0
    assert report["per_query"][0]["token_support"] is None


def test_contracted_abstention_is_detected():
    report = evaluate(
        [GroundingCase("q1", "I don't know.", (Context("a", "No evidence."),))]
    )

    assert report["abstained_count"] == 1
    assert report["metrics"]["mean_token_support"] is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"query_id": "q", "answer": "answer", "contexts": []},
            "non-empty list",
        ),
        (
            {
                "query_id": "q",
                "answer": "answer",
                "contexts": [
                    {"chunk_id": "a", "text": "one"},
                    {"chunk_id": "a", "text": "two"},
                ],
            },
            "duplicate chunk_id",
        ),
        (
            {
                "query_id": "q",
                "answer": "answer",
                "contexts": [{"chunk_id": "a", "text": ""}],
            },
            "non-empty text",
        ),
    ],
)
def test_load_cases_rejects_invalid_traces(tmp_path, payload, message):
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cases(traces)


def test_load_cases_rejects_duplicate_query_ids(tmp_path):
    traces = tmp_path / "traces.jsonl"
    record = {
        "query_id": "q",
        "answer": "Berkeley",
        "contexts": [{"chunk_id": "a", "text": "UC Berkeley"}],
    }
    traces.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate query_id"):
        load_cases(traces)


def test_cli_writes_provenance_and_enforces_thresholds(tmp_path):
    traces = tmp_path / "traces.jsonl"
    report_path = tmp_path / "report.json"
    traces.write_text(
        json.dumps(
            {
                "query_id": "q",
                "answer": "Berkeley lab",
                "contexts": [{"chunk_id": "a", "text": "Berkeley campus"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(traces),
                "--output",
                str(report_path),
                "--minimum",
                "mean_token_support=0.5",
                "--maximum",
                "abstention_rate=0",
            ]
        )
        == 0
    )
    artifact = json.loads(report_path.read_text())
    assert artifact["schema_version"] == 1
    assert len(artifact["input"]["sha256"]) == 64
    assert artifact["thresholds"]["passed"] is True

    assert main([str(traces), "--minimum", "mean_token_support=0.75"]) == 1


def test_cli_rejects_unknown_metric(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "query_id": "q",
                "answer": "Berkeley",
                "contexts": [{"chunk_id": "a", "text": "Berkeley"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown metric"):
        main([str(traces), "--minimum", "faithfulness=0.9"])
