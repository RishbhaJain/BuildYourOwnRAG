"""
Evaluation script for the RAG pipeline.
Computes Exact Match (EM) and token-level F1 between predictions and reference answers.

Usage:
    python3 run_evaluation.py <predictions_file>
"""

import json
import math
import random
import re
import string
import sys
from collections import Counter
from collections.abc import Sequence

REFERENCE_ANSWERS_PATH = "reference_answers.json"
QUESTIONS_PATH = "questions.txt"


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, remove punctuation/articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_confidence_intervals(
    metric_scores: dict[str, Sequence[float]],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 3407,
) -> dict[str, dict[str, float | int]]:
    """Estimate paired bootstrap intervals for per-question mean metrics.

    Every bootstrap replicate resamples the same question indices for all
    metrics. This preserves their per-question relationship and makes the
    result deterministic for a fixed seed.
    """
    if not metric_scores:
        raise ValueError("at least one metric is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if samples < 1:
        raise ValueError("samples must be at least 1")

    lengths = {len(scores) for scores in metric_scores.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all metrics must have the same non-zero length")
    question_count = next(iter(lengths))

    rng = random.Random(seed)
    bootstrap_means = {name: [] for name in metric_scores}
    for _ in range(samples):
        indices = [rng.randrange(question_count) for _ in range(question_count)]
        for name, scores in metric_scores.items():
            bootstrap_means[name].append(
                sum(scores[index] for index in indices) / question_count
            )

    alpha = 1 - confidence
    return {
        name: {
            "mean": sum(scores) / question_count,
            "confidence_interval_low": _percentile(bootstrap_means[name], alpha / 2),
            "confidence_interval_high": _percentile(
                bootstrap_means[name], 1 - alpha / 2
            ),
            "confidence": confidence,
            "bootstrap_samples": samples,
            "seed": seed,
            "question_count": question_count,
        }
        for name, scores in metric_scores.items()
    }


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <predictions_file>")
        sys.exit(1)

    predictions_path = sys.argv[1]

    with open(QUESTIONS_PATH) as f:
        questions = [line.strip() for line in f if line.strip()]

    with open(REFERENCE_ANSWERS_PATH) as f:
        ref_answers = json.load(f)

    with open(predictions_path) as f:
        predictions = [line.strip() for line in f]

    if len(predictions) != len(questions):
        print(
            f"WARNING: {len(predictions)} predictions but {len(questions)} questions. "
            f"Evaluating first {min(len(predictions), len(questions))} entries."
        )

    n = min(len(predictions), len(questions))
    em_scores = []
    f1_scores = []

    print(f"{'#':>3}  {'EM':>4}  {'F1':>6}  {'Prediction':<40}  {'Reference':<40}")
    print("-" * 100)

    for i in range(n):
        ref = ref_answers[str(i)]
        pred = predictions[i]

        # Support multiple ground truth answers (take max score per SQuAD eval)
        # Handles lists and pipe-delimited strings (autograder format)
        if isinstance(ref, str):
            ref = [r.strip() for r in ref.split("|")]
        em = max(exact_match(pred, r) for r in ref)
        f1 = max(token_f1(pred, r) for r in ref)
        ref_display_str = ref[0]
        em_scores.append(em)
        f1_scores.append(f1)

        # truncate display strings
        pred_display = pred[:38] + ".." if len(pred) > 40 else pred
        ref_display = (
            ref_display_str[:38] + ".."
            if len(ref_display_str) > 40
            else ref_display_str
        )

        print(f"{i:>3}  {em:>4.0f}  {f1:>6.3f}  {pred_display:<40}  {ref_display:<40}")

    print("-" * 100)
    intervals = bootstrap_mean_confidence_intervals(
        {"exact_match": em_scores, "token_f1": f1_scores}
    )
    em = intervals["exact_match"]
    f1 = intervals["token_f1"]
    confidence = float(em["confidence"]) * 100
    print(
        f"Exact Match: {float(em['mean']) * 100:.2f}% "
        f"({confidence:.0f}% CI {float(em['confidence_interval_low']) * 100:.2f}% "
        f"to {float(em['confidence_interval_high']) * 100:.2f}%)  |  "
        f"F1: {float(f1['mean']) * 100:.2f}% "
        f"({confidence:.0f}% CI {float(f1['confidence_interval_low']) * 100:.2f}% "
        f"to {float(f1['confidence_interval_high']) * 100:.2f}%)  |  "
        f"n={n}  |  bootstrap samples={int(em['bootstrap_samples'])}  |  "
        f"seed={int(em['seed'])}"
    )


if __name__ == "__main__":
    main()
