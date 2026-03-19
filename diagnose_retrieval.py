"""
Retrieval diagnostic for questions where the answer is known to be in the corpus
but the pipeline output "Unknown".

For each question:
  - Retrieve top-N chunks via dense, BM25, and hybrid
  - Search each chunk for the known answer string
  - Report what rank the answer chunk appeared at (or "not found in top N")
  - Show the actual top-5 chunks so we can see what the model saw

Usage:
    python3 diagnose_retrieval.py
    python3 diagnose_retrieval.py --top-n 50   # search deeper
"""

import argparse
import re
import sys

# ---------------------------------------------------------------------------
# Questions where answer is confirmed in corpus, with known answer strings
# ---------------------------------------------------------------------------
CASES = [
    {
        "question": "When was Dawn Song named a MacArthur Fellow?",
        "answer": "2010",
        # more specific anchor so we don't match random "2010" occurrences
        "corpus_anchor": r"MacArthur.{0,30}Dawn Song|Dawn Song.{0,30}MacArthur",
    },
    {
        "question": "What is Jelani Nelson's email?",
        "answer": "minilek@berkeley.edu",
        "corpus_anchor": r"minilek@berkeley\.edu",
    },
    {
        "question": "What is the technical report number for Jacob Andreas's dissertation?",
        "answer": "UCB/EECS-2018-141",
        "corpus_anchor": r"UCB/EECS-2018-141",
    },
]


def chunk_contains_answer(chunk_text: str, anchor: str) -> bool:
    return bool(re.search(anchor, chunk_text, re.IGNORECASE | re.DOTALL))


def retrieve_and_diagnose(question: str, answer: str, anchor: str,
                          dense, bm25, top_n: int):
    print(f"\n{'='*70}")
    print(f"Q: {question}")
    print(f"Expected answer: {answer!r}")
    print(f"{'='*70}")

    # --- Dense retrieval ---
    dense_results = dense.retrieve_top_k(question, k=top_n)
    dense_hit = next(
        (i + 1 for i, c in enumerate(dense_results)
         if chunk_contains_answer(c["text"], anchor)),
        None,
    )
    print(f"\n[Dense] top-{top_n}: answer chunk at rank {dense_hit or f'>{top_n} (not found)'}")

    # --- BM25 retrieval ---
    bm25_results = bm25.retrieve_top_k(question, k=top_n)
    bm25_hit = next(
        (i + 1 for i, c in enumerate(bm25_results)
         if chunk_contains_answer(c["text"], anchor)),
        None,
    )
    print(f"[BM25]  top-{top_n}: answer chunk at rank {bm25_hit or f'>{top_n} (not found)'}")

    # --- Hybrid (RRF) retrieval ---
    from retriever.fusion import reciprocal_rank_fusion
    hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_n)
    hybrid_hit = next(
        (i + 1 for i, c in enumerate(hybrid_results)
         if chunk_contains_answer(c["text"], anchor)),
        None,
    )
    print(f"[Hybrid] top-{top_n}: answer chunk at rank {hybrid_hit or f'>{top_n} (not found)'}")

    # --- Show top-5 hybrid chunks so we can see what the model got ---
    print(f"\nTop-5 hybrid chunks the model actually saw:")
    for i, chunk in enumerate(hybrid_results[:5], 1):
        has_answer = chunk_contains_answer(chunk["text"], anchor)
        marker = "  <-- ANSWER HERE" if has_answer else ""
        snippet = chunk["text"].replace("\n", " ")[:120]
        print(f"  [{i}] {snippet!r}{marker}")

    # --- If answer wasn't in top 5 but was found deeper, show it ---
    if hybrid_hit and hybrid_hit > 5:
        answer_chunk = hybrid_results[hybrid_hit - 1]
        snippet = answer_chunk["text"].replace("\n", " ")[:200]
        print(f"\n  [rank {hybrid_hit}] (answer chunk, not shown to model):")
        print(f"  {snippet!r}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Retrieval diagnostic for known-answer questions")
    parser.add_argument("--top-n", type=int, default=30,
                        help="How many chunks to retrieve per retriever (default: 30)")
    args = parser.parse_args()

    print(f"Loading retrievers (top-n={args.top_n})...")
    from retriever.dense_retriever import DenseRetriever
    from retriever.bm25_retriever import BM25Retriever

    dense = DenseRetriever()
    dense.load_index()

    bm25 = BM25Retriever()
    bm25.load_bm25()
    print("Retrievers ready.\n")

    for case in CASES:
        retrieve_and_diagnose(
            question=case["question"],
            answer=case["answer"],
            anchor=case["corpus_anchor"],
            dense=dense,
            bm25=bm25,
            top_n=args.top_n,
        )

    print("\nDiagnosis complete.")
    print("Interpretation guide:")
    print("  - Dense rank >> BM25 rank: query is lexically specific, BM25 wins here")
    print("  - Both ranks high (>10): retrieval needs more K, or chunking split the answer")
    print("  - Rank > top_n: answer chunk not reachable at current K — expand K")
    print("  - In top-5 but model said Unknown: generation/reranking issue, not retrieval")


if __name__ == "__main__":
    main()
