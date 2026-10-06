"""Deterministic lexical support diagnostics for RAG generation traces.

The evaluator measures whether the content tokens in each generated answer are
present in the retrieved context. This is a reproducible hallucination signal,
not a semantic entailment judge: high support does not prove that an answer is
factually correct, and paraphrases can be under-counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
ABSTENTIONS = {
    "",
    "unknown",
    "i don't know",
    "i dont know",
    "i do not know",
    "not enough information",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)


@dataclass(frozen=True)
class Context:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class GroundingCase:
    query_id: str
    answer: str
    contexts: tuple[Context, ...]


def tokenize(text: str) -> list[str]:
    """Return stable Unicode-aware word and number tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return TOKEN_PATTERN.findall(normalized)


def content_tokens(text: str) -> list[str]:
    """Remove common function words while preserving non-empty answers."""
    tokens = tokenize(text)
    filtered = [token for token in tokens if token not in STOPWORDS]
    return filtered or tokens


def _is_abstention(answer: str) -> bool:
    normalized = " ".join(tokenize(answer))
    return normalized in ABSTENTIONS


def load_cases(path: str | Path) -> list[GroundingCase]:
    """Load and validate one generation trace per non-empty JSONL line."""
    cases: list[GroundingCase] = []
    seen_query_ids: set[str] = set()

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc

            query_id = item.get("query_id")
            answer = item.get("answer")
            contexts = item.get("contexts")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"line {line_number}: query_id must be a non-empty string"
                )
            if query_id in seen_query_ids:
                raise ValueError(f"line {line_number}: duplicate query_id {query_id!r}")
            if not isinstance(answer, str):
                raise TypeError(f"{query_id}: answer must be a string")
            if not isinstance(contexts, list) or not contexts:
                raise ValueError(f"{query_id}: contexts must be a non-empty list")

            parsed_contexts: list[Context] = []
            seen_chunk_ids: set[str] = set()
            for context_index, context in enumerate(contexts):
                if not isinstance(context, dict):
                    raise TypeError(
                        f"{query_id}: context {context_index} must be an object"
                    )
                chunk_id = context.get("chunk_id")
                text = context.get("text")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(
                        f"{query_id}: context {context_index} needs a non-empty chunk_id"
                    )
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"{query_id}: duplicate chunk_id {chunk_id!r}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{query_id}: context {context_index} needs non-empty text"
                    )
                seen_chunk_ids.add(chunk_id)
                parsed_contexts.append(Context(chunk_id, text))

            seen_query_ids.add(query_id)
            cases.append(GroundingCase(query_id, answer, tuple(parsed_contexts)))

    if not cases:
        raise ValueError("grounding evaluation requires at least one case")
    return cases


def _support_fraction(answer_tokens: Sequence[str], evidence_tokens: set[str]) -> float:
    if not answer_tokens:
        return 0.0
    return sum(token in evidence_tokens for token in answer_tokens) / len(answer_tokens)


def evaluate(cases: Iterable[GroundingCase]) -> dict:
    """Compute aggregate and per-query lexical support diagnostics."""
    per_query: list[dict] = []
    answered_support: list[float] = []
    fully_supported = 0
    numeric_answers = 0
    unsupported_numeric_answers = 0
    abstained = 0

    for case in cases:
        if _is_abstention(case.answer):
            abstained += 1
            per_query.append(
                {
                    "query_id": case.query_id,
                    "abstained": True,
                    "token_support": None,
                    "fully_supported": None,
                    "unsupported_tokens": [],
                    "unsupported_numeric_tokens": [],
                    "supporting_chunk_ids": [],
                }
            )
            continue

        answer_tokens = content_tokens(case.answer)
        context_token_sets = [set(tokenize(context.text)) for context in case.contexts]
        all_context_tokens = set().union(*context_token_sets)
        support = _support_fraction(answer_tokens, all_context_tokens)
        unsupported = sorted(
            {token for token in answer_tokens if token not in all_context_tokens}
        )
        answer_numbers = {
            token for token in answer_tokens if any(c.isdigit() for c in token)
        }
        unsupported_numbers = sorted(answer_numbers - all_context_tokens)
        supporting_chunk_ids = [
            context.chunk_id
            for context, tokens in zip(case.contexts, context_token_sets, strict=True)
            if set(answer_tokens) & tokens
        ]
        max_context_support = max(
            _support_fraction(answer_tokens, tokens) for tokens in context_token_sets
        )

        answered_support.append(support)
        fully_supported += int(not unsupported)
        if answer_numbers:
            numeric_answers += 1
            unsupported_numeric_answers += int(bool(unsupported_numbers))
        per_query.append(
            {
                "query_id": case.query_id,
                "abstained": False,
                "token_support": support,
                "max_context_support": max_context_support,
                "fully_supported": not unsupported,
                "unsupported_tokens": unsupported,
                "unsupported_numeric_tokens": unsupported_numbers,
                "supporting_chunk_ids": supporting_chunk_ids,
            }
        )

    query_count = len(per_query)
    answered_count = len(answered_support)
    return {
        "schema_version": SCHEMA_VERSION,
        "query_count": query_count,
        "answered_count": answered_count,
        "abstained_count": abstained,
        "numeric_answer_count": numeric_answers,
        "metrics": {
            "abstention_rate": abstained / query_count,
            "mean_token_support": (
                sum(answered_support) / answered_count if answered_count else None
            ),
            "fully_supported_answer_rate": (
                fully_supported / answered_count if answered_count else None
            ),
            "unsupported_number_rate": (
                unsupported_numeric_answers / numeric_answers
                if numeric_answers
                else None
            ),
        },
        "per_query": per_query,
        "limitations": (
            "Lexical support is a deterministic diagnostic, not semantic entailment "
            "or proof of factual correctness. Paraphrases may be under-counted."
        ),
    }


def _parse_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        try:
            metric, raw_threshold = value.split("=", 1)
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(
                f"invalid threshold {value!r}; expected metric=value"
            ) from exc
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold for {metric!r} must be between 0 and 1")
        thresholds[metric] = threshold
    return thresholds


def _threshold_failures(
    metrics: dict[str, float | None],
    minimums: dict[str, float],
    maximums: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    for metric, threshold in minimums.items():
        if metric not in metrics:
            raise ValueError(f"unknown metric {metric!r}")
        actual = metrics[metric]
        if actual is None:
            failures.append(f"{metric} is unavailable")
        elif actual < threshold:
            failures.append(f"{metric}={actual:.4f} is below {threshold:.4f}")
    for metric, threshold in maximums.items():
        if metric not in metrics:
            raise ValueError(f"unknown metric {metric!r}")
        actual = metrics[metric]
        if actual is None:
            failures.append(f"{metric} is unavailable")
        elif actual > threshold:
            failures.append(f"{metric}={actual:.4f} is above {threshold:.4f}")
    return failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="JSONL generation traces")
    parser.add_argument("--output", type=Path, help="Write a versioned JSON report")
    parser.add_argument(
        "--minimum",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Fail when a metric is below the threshold",
    )
    parser.add_argument(
        "--maximum",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Fail when a metric is above the threshold",
    )
    args = parser.parse_args(argv)

    report = evaluate(load_cases(args.traces))
    report["input"] = {
        "path": str(args.traces),
        "sha256": _sha256(args.traces),
    }
    minimums = _parse_thresholds(args.minimum)
    maximums = _parse_thresholds(args.maximum)
    failures = _threshold_failures(report["metrics"], minimums, maximums)
    report["thresholds"] = {
        "minimum": minimums,
        "maximum": maximums,
        "passed": not failures,
        "failures": failures,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    metrics = report["metrics"]
    print(f"Queries: {report['query_count']} | answered: {report['answered_count']}")
    for name, value in metrics.items():
        rendered = "n/a" if value is None else f"{value:.2%}"
        print(f"{name}: {rendered}")
    for failure in failures:
        print(f"FAILED: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
