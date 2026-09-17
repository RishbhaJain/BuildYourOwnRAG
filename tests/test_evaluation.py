"""Unit tests for deterministic exact-match and token-F1 metrics."""

import pytest

from run_evaluation import exact_match, normalize, token_f1


def test_normalize_ignores_case_punctuation_articles_and_extra_space():
    assert normalize("  The Master, of Science!  ") == "master of science"


def test_exact_match_uses_normalized_text():
    assert exact_match("An EECS program.", "EECS program") == 1.0


def test_exact_match_rejects_different_answers():
    assert exact_match("dense retrieval", "sparse retrieval") == 0.0


def test_token_f1_is_one_for_an_exact_match():
    assert token_f1("reciprocal rank fusion", "reciprocal rank fusion") == 1.0


def test_token_f1_scores_partial_overlap():
    assert token_f1("machine learning systems", "learning systems") == pytest.approx(0.8)


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
