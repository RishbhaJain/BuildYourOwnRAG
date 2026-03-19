"""
Generator module: takes a question + retrieved passages and produces a short answer
via the provided llm.py wrapper.
"""

import logging

from llm import call_llm
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a factoid QA assistant for UC Berkeley EECS. "
    "Given context passages, answer the question in as few words as possible "
    "(ideally 1-5 words). Output ONLY the answer — no explanations, no punctuation "
    "unless it is part of the answer itself (e.g. an email address or quoted title). "
    "For yes/no questions, answer Yes or No. "
    "For questions asking 'how long ago', you MUST subtract the year from 2026 and "
    "output the result as a number of years (e.g. if the year is 1985, output '41 years'). "
    "Never output a raw year as the answer to a 'how long ago' question. "
    "Only say \"Unknown\" if the context contains no relevant information at all."
)


def format_context(passages: list[dict]) -> str:
    """Format retrieved passages into a numbered context block.

    Each passage dict should have at least a 'text' key,
    and optionally a 'title' key.
    """
    parts = []
    for i, p in enumerate(passages, 1):
        title = p.get("title", "")
        text = p["text"]
        if title:
            parts.append(f"[{i}] {title}\n{text}")
        else:
            parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


def build_query(question: str, passages: list[dict]) -> str:
    """Build the user message combining context and question."""
    context = format_context(passages)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def postprocess_answer(raw: str) -> str:
    """Clean up LLM output to meet assignment requirements:
    - Strip whitespace/newlines
    - Take first line only
    - Truncate to 10 words max
    """
    answer = raw.strip().split("\n")[0].strip()
    words = answer.split()
    if len(words) > 10:
        answer = " ".join(words[:10])
    return answer


def generate_answer(
    question: str,
    passages: list[dict],
    model: str = config.LLM_MODEL,
    max_tokens: int = config.MAX_NEW_TOKENS,
    fallback: str = "Unknown",
) -> str:
    """Generate an answer for a question given retrieved passages.

    Args:
        question: The input question.
        passages: List of passage dicts with 'text' and optional 'title'.
        model: LLM model identifier.
        max_tokens: Max tokens for LLM response.
        fallback: Answer returned on LLM failure.

    Returns:
        A short answer string.
    """
    query = build_query(question, passages)
    try:
        raw = call_llm(
            query=query,
            system_prompt=SYSTEM_PROMPT,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except RuntimeError as e:
        logger.warning("[UNKNOWN-REASON: api_error] question=%r error=%s", question[:60], e)
        return fallback

    answer = postprocess_answer(raw)

    if not answer:
        logger.warning("[UNKNOWN-REASON: empty_response] question=%r raw=%r", question[:60], raw[:50])
        return fallback

    if answer.lower() == "unknown":
        logger.info("[UNKNOWN-REASON: llm_said_unknown] question=%r", question[:60])

    return answer
