"""
End-to-end RAG pipeline: questions → retrieval → generation → predictions.

Usage:
    python3 run_pipeline.py <questions_path> <predictions_path>
    python3 run_pipeline.py questions.txt predictions.txt
    python3 run_pipeline.py questions.txt predictions.txt --retriever hybrid --top-k 5
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import config
from retriever.dense_retriever import DenseRetriever
from retriever.bm25_retriever import BM25Retriever
from llms.llm_pipeline import generate_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_predictions(path: str, predictions: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Combine dense and BM25 results using reciprocal rank fusion."""
    scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk_map[cid] for cid, _ in ranked]


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline: questions → predictions")
    parser.add_argument("questions_path", help="Path to input questions file (one per line)")
    parser.add_argument("predictions_path", help="Path to write predictions (one per line)")
    parser.add_argument("--retriever", default="hybrid", choices=["dense", "hybrid"],
                        help="Retrieval method (default: hybrid)")
    parser.add_argument("--top-k", type=int, default=config.DENSE_TOP_K,
                        help="Number of passages to retrieve per question")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent LLM workers")
    args = parser.parse_args()

    questions = load_questions(args.questions_path)
    logger.info("Loaded %d questions from %s", len(questions), args.questions_path)

    # --- Build retrievers ---
    logger.info("Building %s retriever...", args.retriever)
    dense = DenseRetriever()
    dense.load_index()

    bm25 = None
    if args.retriever == "hybrid":
        bm25 = BM25Retriever()
        bm25.load_bm25()

    logger.info("Retriever ready.")

    # --- Batch retrieval: encode all queries at once for dense ---
    t0 = time.time()
    logger.info("Batch-encoding %d queries for dense retrieval...", len(questions))
    dense_results_all = dense.batch_retrieve_top_k(questions, k=args.top_k)
    logger.info("Dense retrieval done in %.1fs", time.time() - t0)

    # Combine with BM25 if hybrid
    if bm25 is not None:
        logger.info("Running BM25 retrieval...")
        t_bm25 = time.time()
        passages_all = []
        for i, question in enumerate(questions):
            bm25_results = bm25.retrieve_top_k(question, k=args.top_k)
            fused = reciprocal_rank_fusion(dense_results_all[i], bm25_results, top_k=args.top_k)
            passages_all.append(fused)
        logger.info("Hybrid retrieval done in %.1fs", time.time() - t_bm25)
    else:
        passages_all = dense_results_all

    # --- Parallel LLM generation ---
    logger.info("Generating answers with %d concurrent workers...", args.workers)
    predictions = ["Unknown"] * len(questions)
    t_gen = time.time()

    def generate_one(idx):
        try:
            return idx, generate_answer(questions[idx], passages_all[idx])
        except Exception as e:
            logger.warning("Question %d failed: %s — using fallback", idx, e)
            return idx, "Unknown"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_one, i): i for i in range(len(questions))}
        done_count = 0
        for future in as_completed(futures):
            idx, answer = future.result()
            predictions[idx] = answer
            done_count += 1
            if done_count % 20 == 0 or done_count == len(questions):
                elapsed = time.time() - t_gen
                logger.info("[%d/%d] generation %.1fs elapsed", done_count, len(questions), elapsed)

    write_predictions(args.predictions_path, predictions)
    logger.info("Wrote %d predictions to %s", len(predictions), args.predictions_path)

    logger.info("=== Per-question predictions ===")
    for i, (q, pred) in enumerate(zip(questions, predictions)):
        logger.info("[%d] Q: %s | A: %s", i + 1, q, pred)

    total = time.time() - t0
    logger.info("Done in %.1fs (%.2fs/question avg)", total, total / len(questions))


if __name__ == "__main__":
    main()
