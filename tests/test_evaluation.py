"""Unit tests for deterministic exact-match and token-F1 metrics."""

import pytest

from run_evaluation import (
    bootstrap_mean_confidence_intervals,
    exact_match,
    normalize,
    token_f1,
)


def test_normalize_ignores_case_punctuation_articles_and_extra_space():
    assert normalize("  The Master, of Science!  ") == "master of science"


def test_exact_match_uses_normalized_text():
    assert exact_match("An EECS program.", "EECS program") == 1.0


def test_exact_match_rejects_different_answers():
    assert exact_match("dense retrieval", "sparse retrieval") == 0.0


def test_token_f1_is_one_for_an_exact_match():
    assert token_f1("reciprocal rank fusion", "reciprocal rank fusion") == 1.0


def test_token_f1_scores_partial_overlap():
    assert token_f1("machine learning systems", "learning systems") == pytest.approx(
        0.8
    )


def test_token_f1_counts_duplicate_tokens_correctly():
    assert token_f1("model model model", "model model") == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("", "", 1.0),
        ("", "answer", 0.0),
        ("answer", "", 0.0),
        ("dense", "sparse", 0.0),
    ],
)
def test_token_f1_handles_empty_and_disjoint_answers(prediction, reference, expected):
    assert token_f1(prediction, reference) == expected


def test_bootstrap_confidence_intervals_are_deterministic_and_paired():
    scores = {
        "exact_match": [1.0, 0.0, 1.0, 0.0],
        "token_f1": [1.0, 0.25, 0.75, 0.0],
    }

    first = bootstrap_mean_confidence_intervals(scores, samples=500, seed=7)
    second = bootstrap_mean_confidence_intervals(scores, samples=500, seed=7)

    assert first == second
    assert first["exact_match"]["mean"] == 0.5
    assert first["token_f1"]["mean"] == 0.5
    assert first["exact_match"]["question_count"] == 4
    assert first["exact_match"]["bootstrap_samples"] == 500
    assert first["exact_match"]["confidence_interval_low"] <= 0.5
    assert first["exact_match"]["confidence_interval_high"] >= 0.5


def test_bootstrap_interval_collapses_for_constant_scores():
    result = bootstrap_mean_confidence_intervals(
        {"exact_match": [1.0, 1.0, 1.0]}, samples=50
    )["exact_match"]

    assert result["mean"] == 1.0
    assert result["confidence_interval_low"] == 1.0
    assert result["confidence_interval_high"] == 1.0


@pytest.mark.parametrize(
    ("scores", "kwargs", "message"),
    [
        ({}, {}, "at least one metric"),
        ({"em": []}, {}, "same non-zero length"),
        ({"em": [1.0], "f1": [1.0, 0.0]}, {}, "same non-zero length"),
        ({"em": [1.0]}, {"confidence": 1.0}, "between 0 and 1"),
        ({"em": [1.0]}, {"samples": 0}, "at least 1"),
    ],
)
def test_bootstrap_confidence_intervals_reject_invalid_inputs(scores, kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_mean_confidence_intervals(scores, **kwargs)
