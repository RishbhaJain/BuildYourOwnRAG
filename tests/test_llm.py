"""Tests for the generator module (llms/llm_pipeline.py).

These tests mock call_llm so they run without an API key.
"""

from unittest.mock import patch

import pytest

from llm import LLMResponse
from llms.llm_pipeline import (
    GenerationResult,
    ProviderGenerationError,
    build_query,
    format_context,
    generate_answer,
    generate_answer_with_metrics,
    postprocess_answer,
)

# --- format_context ---


def test_format_context_with_titles():
    passages = [
        {"title": "Page A", "text": "Some content."},
        {"title": "Page B", "text": "Other content."},
    ]
    result = format_context(passages)
    assert "[1] Page A\nSome content." in result
    assert "[2] Page B\nOther content." in result


def test_format_context_without_titles():
    passages = [{"text": "Just text."}]
    result = format_context(passages)
    assert "[1] Just text." in result
    assert "\n" not in result.split("[1] ")[1]  # no title line


# --- build_query ---


def test_build_query_structure():
    passages = [{"title": "T", "text": "passage text"}]
    result = build_query("What is X?", passages)
    assert result.startswith("Context:")
    assert "Question: What is X?" in result
    assert result.endswith("Answer:")


# --- postprocess_answer ---


def test_postprocess_strips_whitespace():
    assert postprocess_answer("  hello world  \n") == "hello world"


def test_postprocess_takes_first_line():
    assert postprocess_answer("answer\nexplanation here") == "answer"


def test_postprocess_truncates_long_answers():
    long = " ".join(["word"] * 15)
    result = postprocess_answer(long)
    assert len(result.split()) == 10


def test_postprocess_empty_string():
    assert postprocess_answer("") == ""


# --- generate_answer ---


@patch("llms.llm_pipeline.call_llm")
def test_generate_answer_success(mock_llm):
    mock_llm.return_value = "1965"
    passages = [{"title": "History", "text": "Founded in 1965."}]
    answer = generate_answer("When was it founded?", passages)
    assert answer == "1965"
    mock_llm.assert_called_once()


@patch("llms.llm_pipeline.call_llm")
def test_generate_answer_timeout_returns_fallback(mock_llm):
    mock_llm.side_effect = RuntimeError("OpenRouter request timed out")
    passages = [{"text": "some text"}]
    answer = generate_answer("question?", passages)
    assert answer == "Unknown"


@patch("llms.llm_pipeline.call_llm")
def test_generate_answer_empty_response_returns_fallback(mock_llm):
    mock_llm.return_value = "   "
    passages = [{"text": "some text"}]
    answer = generate_answer("question?", passages)
    assert answer == "Unknown"


@patch("llms.llm_pipeline.call_llm_with_metrics")
def test_generate_answer_with_metrics_preserves_provider_telemetry(mock_llm):
    mock_llm.return_value = LLMResponse(
        content="  Dan Garcia\nextra explanation",
        prompt_tokens=120,
        completion_tokens=4,
        total_tokens=124,
        cost_usd=0.00042,
        ttft_seconds=0.18,
    )

    result = generate_answer_with_metrics(
        "Who leads the group?",
        [{"title": "People", "text": "Dan Garcia leads the group."}],
    )

    assert result == GenerationResult(
        answer="Dan Garcia",
        prompt_tokens=120,
        completion_tokens=4,
        total_tokens=124,
        cost_usd=0.00042,
        ttft_seconds=0.18,
    )


@patch("llms.llm_pipeline.call_llm_with_metrics")
def test_generate_answer_with_metrics_propagates_provider_failure(mock_llm):
    mock_llm.side_effect = RuntimeError("OpenRouter request timed out")

    with pytest.raises(ProviderGenerationError, match="provider request failed"):
        generate_answer_with_metrics("question?", [{"text": "some text"}])
