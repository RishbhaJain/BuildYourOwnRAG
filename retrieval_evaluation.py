"""Deterministic ranking metrics for retrieval evaluation artifacts.

The evaluator is intentionally independent of a model or vector database.  It
consumes JSONL traces containing the relevant and retrieved chunk identifiers,
which makes dense, lexical, hybrid, and future retrievers directly comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCase:
    query_id: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]


def _require_unique(values: Sequence[str], field: str, query_id: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{query_id}: {field} must not contain duplicates")


def load_cases(path: str | Path) -> list[RetrievalCase]:
    """Load and validate one retrieval case per non-empty JSONL line."""
    cases: list[RetrievalCase] = []
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
            relevant = item.get("relevant_chunk_ids")
            retrieved = item.get("retrieved_chunk_ids")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"line {line_number}: query_id must be a non-empty string"
                )
            if query_id in seen_query_ids:
                raise ValueError(f"line {line_number}: duplicate query_id {query_id!r}")
            if not isinstance(relevant, list):
                raise TypeError(f"{query_id}: relevant_chunk_ids must be a list")
            if not relevant:
                raise ValueError(f"{query_id}: relevant_chunk_ids must not be empty")
            if not isinstance(retrieved, list):
                raise TypeError(f"{query_id}: retrieved_chunk_ids must be a list")
            if not all(isinstance(value, str) and value for value in relevant):
                raise ValueError(f"{query_id}: relevant_chunk_ids must contain strings")
            if not all(isinstance(value, str) and value for value in retrieved):
                raise ValueError(
                    f"{query_id}: retrieved_chunk_ids must contain strings"
                )

            _require_unique(relevant, "relevant_chunk_ids", query_id)
            _require_unique(retrieved, "retrieved_chunk_ids", query_id)
            seen_query_ids.add(query_id)
            cases.append(RetrievalCase(query_id, tuple(relevant), tuple(retrieved)))

    if not cases:
        raise ValueError("retrieval evaluation requires at least one case")
    return cases


def _metrics_at_k(case: RetrievalCase, k: int) -> dict[str, float]:
    relevant = set(case.relevant_chunk_ids)
    ranked = case.retrieved_chunk_ids[:k]
    hit_ranks = [
        rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant
    ]
    recall = len(hit_ranks) / len(relevant)
    reciprocal_rank = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "hit_rate": float(bool(hit_ranks)),
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": dcg / ideal_dcg,
    }


def evaluate(cases: Iterable[RetrievalCase], ks: Sequence[int]) -> dict:
    """Return macro-averaged metrics and query-level diagnostics."""
    cases = list(cases)
    normalized_ks = sorted(set(ks))
    if not cases:
        raise ValueError("retrieval evaluation requires at least one case")
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain positive integers")

    per_query = []
    totals = {
        f"{name}@{k}": 0.0
        for k in normalized_ks
        for name in ("hit_rate", "recall", "mrr", "ndcg")
    }
    for case in cases:
        query_metrics: dict[str, float] = {}
        for k in normalized_ks:
            for name, value in _metrics_at_k(case, k).items():
                metric = f"{name}@{k}"
                query_metrics[metric] = value
                totals[metric] += value
        per_query.append(
            {
                "query_id": case.query_id,
                "relevant_count": len(case.relevant_chunk_ids),
                "retrieved_count": len(case.retrieved_chunk_ids),
                "metrics": query_metrics,
            }
        )

    aggregate = {metric: value / len(cases) for metric, value in totals.items()}
    return {
        "schema_version": 1,
        "query_count": len(cases),
        "ks": normalized_ks,
        "metrics": aggregate,
        "per_query": per_query,
    }


def parse_threshold(value: str) -> tuple[str, float]:
    try:
        metric, raw_threshold = value.split("=", maxsplit=1)
        threshold = float(raw_threshold)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "threshold must use metric@k=value, for example recall@5=0.80"
        ) from exc
    if not metric or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold value must be between 0 and 1")
    return metric, threshold


def parse_ks(value: str) -> list[int]:
    try:
        ks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ks must be comma-separated integers") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("ks must contain positive integers")
    return ks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL retrieval run to evaluate")
    parser.add_argument("--ks", type=parse_ks, default=[1, 5, 10])
    parser.add_argument("--output", type=Path, help="optional JSON result artifact")
    parser.add_argument(
        "--minimum",
        action="append",
        default=[],
        type=parse_threshold,
        metavar="METRIC=VALUE",
        help="fail when a metric is below a regression threshold (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate(load_cases(args.input), args.ks)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    failures = []
    for metric, threshold in args.minimum:
        if metric not in report["metrics"]:
            available = ", ".join(sorted(report["metrics"]))
            raise ValueError(
                f"unknown metric {metric!r}; available metrics: {available}"
            )
        actual = report["metrics"][metric]
        if actual < threshold:
            failures.append(f"{metric}={actual:.4f} is below {threshold:.4f}")
    if failures:
        print("Retrieval regression gate failed: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
