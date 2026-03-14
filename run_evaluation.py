"""
Evaluation script for the RAG pipeline.
Computes Exact Match (EM) and token-level F1 between predictions and reference answers.

Usage:
    python3 run_evaluation.py <predictions_file>
"""

import json
import re
import string
import sys
from collections import Counter

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
        if isinstance(ref, list):
            em = max(exact_match(pred, r) for r in ref)
            f1 = max(token_f1(pred, r) for r in ref)
            ref_display_str = ref[0]
        else:
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)
            ref_display_str = ref
        em_scores.append(em)
        f1_scores.append(f1)

        # truncate display strings
        pred_display = pred[:38] + ".." if len(pred) > 40 else pred
        ref_display = ref_display_str[:38] + ".." if len(ref_display_str) > 40 else ref_display_str

        print(f"{i:>3}  {em:>4.0f}  {f1:>6.3f}  {pred_display:<40}  {ref_display:<40}")

    print("-" * 100)
    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    print(f"Exact Match: {avg_em:.2f}%  |  F1: {avg_f1:.2f}%  |  n={n}")


if __name__ == "__main__":
    main()
