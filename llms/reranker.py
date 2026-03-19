"""
LLM-based pointwise reranker.

Scores each retrieved passage independently (0-10) and returns passages
sorted by descending relevance score, truncated to top_k.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from llm import call_llm

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = (
    "You are a relevance-scoring assistant. "
    "Given a question and a passage, output ONLY a single integer from 0 to 10 "
    "representing how relevant the passage is to answering the question. "
    "10 = directly answers the question. 0 = completely irrelevant. "
    "Output the integer only. No explanation."
)

SCORE_FALLBACK = 0.0


def _build_score_prompt(question: str, passage_text: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Passage: {passage_text}\n\n"
        "Relevance score (0-10):"
    )


def _parse_score(response: str) -> float:
    """Extract and clamp a numeric score from an LLM response.

    Returns SCORE_FALLBACK (0.0) if no number is found.
    """
    if not response:
        return SCORE_FALLBACK
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', response.strip())
    if match is None:
        logger.warning("Could not parse score from reranker response: %r", response)
        return SCORE_FALLBACK
    return max(0.0, min(10.0, float(match.group(1))))


def _score_passage(question: str, passage: dict, model: str) -> tuple[dict, float]:
    """Score a single passage for relevance to the question.

    On LLM failure, returns (passage, SCORE_FALLBACK) so the passage is ranked
    last rather than dropped — reranking never reduces recall to zero.
    """
    prompt = _build_score_prompt(question, passage.get("text", ""))
    try:
        response = call_llm(
            query=prompt,
            system_prompt=RERANK_SYSTEM_PROMPT,
            model=model,
            max_tokens=8,       # Score is at most "10" — 2 chars; keep calls cheap
            temperature=0.0,
        )
        return passage, _parse_score(response)
    except RuntimeError as e:
        logger.warning(
            "Reranker LLM call failed for passage %r: %s",
            passage.get("chunk_id", "?"), e,
        )
        return passage, SCORE_FALLBACK


def rerank_passages(
    question: str,
    passages: list[dict],
    top_k: int = config.RERANK_TOP_K,
    model: str = config.LLM_MODEL,
    max_workers: int = 5,
) -> list[dict]:
    """Rerank passages by LLM relevance score, return top_k.

    Scores all passages in parallel, sorts by descending score, and returns
    the top_k. Falls back to the original passage list (truncated) on any
    unexpected error so generation is never blocked.

    Args:
        question:    The input question.
        passages:    Candidate passages (each a dict with at least 'text').
        top_k:       How many top passages to return after reranking.
        model:       LLM model identifier (must be in llm.ALLOWED_MODELS).
        max_workers: Thread concurrency for parallel scoring calls.

    Returns:
        Up to top_k passages sorted by descending relevance score.
    """
    if not passages:
        return []

    try:
        scored: list[tuple[dict, float]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_score_passage, question, p, model): p
                for p in passages
            }
            for future in as_completed(futures):
                scored.append(future.result())

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    except Exception as e:
        logger.error("Reranker failed unexpectedly, returning original passages: %s", e)
        return passages[:top_k]
