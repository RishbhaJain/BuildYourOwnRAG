"""Tests for llms/reranker.py.

All LLM calls are mocked — no API key required.
Mock path: "llms.reranker.call_llm" (same convention as test_llm.py).
"""

from unittest.mock import patch
from llms.reranker import _parse_score, _score_passage, rerank_passages, SCORE_FALLBACK


# --- _parse_score ---

def test_parse_score_plain_integer():
    assert _parse_score("7") == 7.0

def test_parse_score_trailing_text():
    assert _parse_score("8 out of 10") == 8.0

def test_parse_score_decimal():
    assert _parse_score("7.5") == 7.5

def test_parse_score_clamps_above_10():
    assert _parse_score("15") == 10.0

def test_parse_score_zero():
    assert _parse_score("0") == 0.0

def test_parse_score_ten():
    assert _parse_score("10") == 10.0

def test_parse_score_empty_string_returns_fallback():
    assert _parse_score("") == SCORE_FALLBACK

def test_parse_score_no_number_returns_fallback():
    assert _parse_score("completely irrelevant passage") == SCORE_FALLBACK

def test_parse_score_bold_markdown_still_parsed():
    # Models sometimes wrap answers in markdown bold
    assert _parse_score("**8**") == 8.0


# --- _score_passage ---

@patch("llms.reranker.call_llm", return_value="8")
def test_score_passage_returns_passage_and_score(mock_llm):
    p = {"chunk_id": "c1", "text": "Berkeley EECS was founded in 1868."}
    passage, score = _score_passage("When was EECS founded?", p, "meta-llama/llama-3.1-8b-instruct")
    assert passage is p
    assert score == 8.0

@patch("llms.reranker.call_llm", side_effect=RuntimeError("timed out"))
def test_score_passage_llm_failure_returns_fallback_not_raises(mock_llm):
    p = {"chunk_id": "c2", "text": "Some text."}
    passage, score = _score_passage("question?", p, "meta-llama/llama-3.1-8b-instruct")
    assert passage is p
    assert score == SCORE_FALLBACK

@patch("llms.reranker.call_llm", return_value="not a number at all")
def test_score_passage_unparseable_response_returns_fallback(mock_llm):
    p = {"chunk_id": "c3", "text": "Some text."}
    _, score = _score_passage("question?", p, "meta-llama/llama-3.1-8b-instruct")
    assert score == SCORE_FALLBACK

@patch("llms.reranker.call_llm", return_value="5")
def test_score_passage_missing_text_key_does_not_raise(mock_llm):
    # Passage dict missing 'text' — should use empty string, not crash
    p = {"chunk_id": "c4", "title": "Some Title"}
    passage, score = _score_passage("question?", p, "meta-llama/llama-3.1-8b-instruct")
    assert passage is p
    assert score == 5.0


# --- rerank_passages ---

def _make_passages(n: int) -> list[dict]:
    return [{"chunk_id": f"c{i}", "text": f"Passage {i} text."} for i in range(n)]


@patch("llms.reranker.call_llm")
def test_rerank_passages_returns_top_k_sorted_descending(mock_llm):
    # passage 0 → score 9, passage 1 → score 3, passage 2 → score 7
    score_map = {"Passage 0": "9", "Passage 1": "3", "Passage 2": "7"}
    def score_by_content(query, **kwargs):
        for key, score in score_map.items():
            if key in query:
                return score
        return "0"
    mock_llm.side_effect = score_by_content
    passages = _make_passages(3)
    result = rerank_passages("question?", passages, top_k=2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c0"   # score 9
    assert result[1]["chunk_id"] == "c2"   # score 7

@patch("llms.reranker.call_llm")
def test_rerank_passages_full_sort_order(mock_llm):
    # Use a side_effect function so scores are tied to passage content,
    # not to the non-deterministic call order from ThreadPoolExecutor.
    score_map = {"Passage 0": "1", "Passage 1": "10", "Passage 2": "5"}
    def score_by_content(query, **kwargs):
        for key, score in score_map.items():
            if key in query:
                return score
        return "0"
    mock_llm.side_effect = score_by_content
    passages = _make_passages(3)
    result = rerank_passages("q?", passages, top_k=3)
    assert [p["chunk_id"] for p in result] == ["c1", "c2", "c0"]  # 10, 5, 1

def test_rerank_passages_empty_input_returns_empty():
    result = rerank_passages("q?", [])
    assert result == []

@patch("llms.reranker.call_llm", return_value="5")
def test_rerank_passages_top_k_larger_than_passages_returns_all(mock_llm):
    passages = _make_passages(3)
    result = rerank_passages("q?", passages, top_k=10)
    assert len(result) == 3

@patch("llms.reranker.call_llm", side_effect=RuntimeError("fail"))
def test_rerank_passages_all_llm_failures_returns_top_k_not_crash(mock_llm):
    # All passages get SCORE_FALLBACK — order is non-deterministic among ties,
    # but we must get exactly top_k back without raising.
    passages = _make_passages(5)
    result = rerank_passages("q?", passages, top_k=3)
    assert len(result) == 3

@patch("llms.reranker.call_llm", return_value="6")
def test_rerank_passages_single_passage(mock_llm):
    passages = [{"chunk_id": "c0", "text": "Only passage."}]
    result = rerank_passages("q?", passages, top_k=5)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "c0"

@patch("llms.reranker.call_llm", return_value="5")
def test_rerank_passages_uses_config_top_k_by_default(mock_llm):
    import config
    passages = _make_passages(config.RERANK_TOP_K + 3)
    result = rerank_passages("q?", passages)
    assert len(result) == config.RERANK_TOP_K
