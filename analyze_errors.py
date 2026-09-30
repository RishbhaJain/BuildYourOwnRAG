"""Attribute RAG answer failures to corpus, retrieval, and generation stages."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from llm import call_llm
from retriever.fusion import reciprocal_rank_fusion
from run_evaluation import exact_match, normalize, token_f1

CORRECT = "CORRECT"
CORPUS_GAP = "CORPUS_GAP"
RETRIEVAL_MISS = "RETRIEVAL_MISS"
FALLBACK = "FALLBACK"
FORMAT_MISMATCH = "FORMAT_MISMATCH"
GENERATION_FAIL = "GENERATION_FAIL"
UNEVALUATED = "UNEVALUATED"

CATEGORY_ORDER = (
    CORRECT,
    CORPUS_GAP,
    RETRIEVAL_MISS,
    FALLBACK,
    FORMAT_MISMATCH,
    GENERATION_FAIL,
    UNEVALUATED,
)


class DenseRetrieverLike(Protocol):
    def batch_retrieve_top_k(self, queries: list[str], k: int) -> list[list[dict]]: ...


class SparseRetrieverLike(Protocol):
    def retrieve_top_k(self, query: str, k: int) -> list[dict]: ...


def load_reference_answers(path: str | Path) -> dict[str, list[str]]:
    """Load references and normalize both supported formats to lists."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    references = {}
    for key, value in raw.items():
        if isinstance(value, str):
            references[str(key)] = [item.strip() for item in value.split("|")]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            references[str(key)] = value
        else:
            raise ValueError(f"reference {key} must be a string or list of strings")
    return references


def compute_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    """Return the best normalized exact match and token F1 over references."""
    if not references:
        return 0.0, 0.0
    return (
        max(exact_match(prediction, reference) for reference in references),
        max(token_f1(prediction, reference) for reference in references),
    )


def _normalize_for_search(text: str) -> str:
    normalized = normalize(text)
    return re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", normalized)


def build_normalized_corpus(chunks: list[dict]) -> list[str]:
    """Pre-normalize chunk text once for repeated reference lookups."""
    return [_normalize_for_search(str(chunk.get("text", ""))) for chunk in chunks]


def _match_reference(references: list[str], normalized_text: str) -> str | None:
    if not normalized_text:
        return None
    for reference in references:
        normalized_reference = _normalize_for_search(reference)
        if not normalized_reference:
            warnings.warn(
                "reference is empty after normalization; skipping containment check",
                stacklevel=2,
            )
            continue
        if normalized_reference in normalized_text:
            return reference
    return None


def is_answer_in_corpus(
    references: list[str], normalized_corpus: list[str]
) -> tuple[bool, str | None, int | None]:
    """Find the first corpus chunk containing any normalized reference."""
    for index, normalized_text in enumerate(normalized_corpus):
        reference = _match_reference(references, normalized_text)
        if reference is not None:
            return True, reference, index
    return False, None, None


def is_answer_in_retrieved(
    references: list[str], retrieved: list[dict]
) -> tuple[bool, str | None, int | None]:
    """Find a reference in retrieved chunks and return its zero-based rank."""
    for rank, chunk in enumerate(retrieved):
        reference = _match_reference(
            references, _normalize_for_search(str(chunk.get("text", "")))
        )
        if reference is not None:
            return True, reference, rank
    return False, None, None


def _best_partial_score(references: list[str], texts: list[str]) -> float:
    return max(
        (token_f1(reference, text) for reference in references for text in texts),
        default=0.0,
    )


def _reference_alias_match(prediction: str, references: list[str]) -> bool:
    """Catch surface-form mismatches such as 'M.S.' vs 'Master of Science'."""
    normalized_prediction = normalize(prediction)
    prediction_tokens = set(normalized_prediction.split())
    for reference in references:
        words = [
            word for word in normalize(reference).split() if word not in {"of", "and"}
        ]
        acronym = "".join(word[0] for word in words if word)
        if len(acronym) >= 2 and acronym in prediction_tokens:
            return True
    return False


def _retrieve(
    question: str,
    dense: DenseRetrieverLike,
    bm25: SparseRetrieverLike | None,
    top_k: int,
) -> list[dict]:
    dense_results = dense.batch_retrieve_top_k([question], k=top_k)[0]
    if bm25 is None:
        return dense_results
    sparse_results = bm25.retrieve_top_k(question, k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)


def categorize(
    *,
    idx: int,
    question: str,
    references: list[str],
    prediction: str,
    norm_corpus: list[str],
    chunks: list[dict],
    dense: DenseRetrieverLike,
    bm25: SparseRetrieverLike | None,
    top_k: int,
) -> dict[str, Any]:
    """Attribute one prediction to the earliest failed RAG stage."""
    em, f1 = compute_scores(prediction, references)
    base = {
        "idx": idx,
        "question": question,
        "references": references,
        "prediction": prediction,
        "em": em,
        "f1": f1,
    }
    if em == 1.0:
        return {
            **base,
            "category": CORRECT,
            "category_reason": "Prediction exactly matches a normalized reference.",
        }

    corpus_found, _, _ = is_answer_in_corpus(references, norm_corpus)
    if not corpus_found:
        return {
            **base,
            "category": CORPUS_GAP,
            "category_reason": "No reference answer appears in the indexed corpus.",
            "best_partial_f1_in_corpus": _best_partial_score(references, norm_corpus),
        }

    retrieved = _retrieve(question, dense, bm25, top_k)
    answer_retrieved, _, answer_rank = is_answer_in_retrieved(references, retrieved)
    if not answer_retrieved:
        answer_chunk_ids = [
            str(chunks[index].get("chunk_id", index))
            for index, text in enumerate(norm_corpus)
            if _match_reference(references, text) is not None
        ]
        expanded = dense.batch_retrieve_top_k([question], k=max(100, top_k))[0]
        _, _, expanded_rank = is_answer_in_retrieved(references, expanded)
        return {
            **base,
            "category": RETRIEVAL_MISS,
            "category_reason": f"Answer evidence was absent from the top-{top_k} context.",
            "answer_chunk_ids": answer_chunk_ids,
            "best_dense_rank": expanded_rank,
            "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in retrieved],
        }

    if not prediction.strip() or normalize(prediction) == "unknown":
        return {
            **base,
            "category": FALLBACK,
            "category_reason": "Answer evidence was retrieved, but generation returned a fallback.",
            "raw_prediction": prediction,
            "answer_rank": answer_rank,
        }

    if f1 > 0 or _reference_alias_match(prediction, references):
        return {
            **base,
            "category": FORMAT_MISMATCH,
            "category_reason": "Prediction overlaps a reference but misses normalized exact match.",
            "norm_prediction": normalize(prediction),
            "norm_reference": normalize(references[0]) if references else "",
            "answer_rank": answer_rank,
        }

    return {
        **base,
        "category": GENERATION_FAIL,
        "category_reason": "Answer evidence was retrieved, but generation produced a wrong answer.",
        "context_f1": _best_partial_score(
            [prediction], [str(chunk.get("text", "")) for chunk in retrieved]
        ),
        "answer_rank": answer_rank,
    }


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def run_llm_judge(results: list[dict[str, Any]]) -> None:
    """Optionally adjudicate lexical mismatches without changing core categories."""
    candidates = [
        result
        for result in results
        if result.get("category") in {FORMAT_MISMATCH, GENERATION_FAIL}
    ]
    if not candidates:
        return
    payload = [
        {
            "idx": result["idx"],
            "question": result["question"],
            "references": result["references"],
            "prediction": result["prediction"],
        }
        for result in candidates
    ]
    query = (
        "Judge whether each prediction is factually equivalent to any reference. "
        "Return only a JSON array of objects with idx, correct, and reason.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        response = call_llm(
            query=query,
            system_prompt="You are a strict factual QA evaluator.",
            max_tokens=2_000,
            temperature=0.0,
        )
        verdicts = json.loads(_strip_code_fence(response))
        if not isinstance(verdicts, list):
            raise TypeError("judge response must be a list")
    except (RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        return

    by_index = {
        verdict.get("idx"): verdict
        for verdict in verdicts
        if isinstance(verdict, dict) and "idx" in verdict
    }
    for result in candidates:
        verdict = by_index.get(result["idx"])
        if verdict is None:
            result["llm_correct"] = None
            continue
        result["llm_correct"] = bool(verdict.get("correct"))
        result["llm_judge_reason"] = str(verdict.get("reason", ""))


def build_summary_table(results: list[dict[str, Any]]) -> str:
    """Render category counts and rates as a compact Markdown table."""
    counts = Counter(str(result.get("category", UNEVALUATED)) for result in results)
    total = len(results)
    rows = ["| Category | Count | Rate |", "|---|---:|---:|"]
    for category in CATEGORY_ORDER:
        count = counts[category]
        if count:
            rate = count / total if total else 0.0
            rows.append(f"| {category} | {count} | {rate:.1%} |")
    return "\n".join(rows)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("questions.txt"))
    parser.add_argument(
        "--references", type=Path, default=Path("reference_answers.json")
    )
    parser.add_argument("--predictions", type=Path, default=Path("predictions.txt"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/error_analysis.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be at least one")

    from retriever.bm25_retriever import BM25Retriever
    from retriever.dense_retriever import DenseRetriever

    questions = [
        line.strip() for line in args.questions.read_text().splitlines() if line.strip()
    ]
    predictions = args.predictions.read_text().splitlines()
    references = load_reference_answers(args.references)
    chunks = _load_jsonl(args.chunks)
    normalized_corpus = build_normalized_corpus(chunks)
    dense = DenseRetriever(chunks_path=str(args.chunks))
    dense.load_index()
    bm25 = (
        BM25Retriever(chunks_path=str(args.chunks))
        if args.retriever == "hybrid"
        else None
    )
    if bm25 is not None:
        bm25.load_bm25()

    results = []
    for index, question in enumerate(questions):
        refs = references.get(str(index), [])
        if index >= len(predictions):
            results.append(
                {
                    "idx": index,
                    "question": question,
                    "references": refs,
                    "prediction": "",
                    "em": 0.0,
                    "f1": 0.0,
                    "category": UNEVALUATED,
                    "category_reason": "No prediction available for this question.",
                }
            )
            continue
        results.append(
            categorize(
                idx=index,
                question=question,
                references=refs,
                prediction=predictions[index],
                norm_corpus=normalized_corpus,
                chunks=chunks,
                dense=dense,
                bm25=bm25,
                top_k=args.top_k,
            )
        )

    if args.llm_judge:
        run_llm_judge(results)
    artifact = {
        "summary": dict(Counter(row["category"] for row in results)),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(build_summary_table(results))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
