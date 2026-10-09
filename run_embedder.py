"""
CLI script to embed all chunks and build the FAISS index.

Usage:
    python3 run_embedder.py
    python3 run_embedder.py --chunks data/chunks.jsonl --embeddings data/embeddings.npy --index data/faiss_index.bin
"""

import argparse
import logging
import time

import config
from retriever.dense_retriever import DenseRetriever

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings and FAISS index from chunks."
    )
    parser.add_argument(
        "--chunks", default=config.CHUNKS_JSONL_PATH, help="Path to chunks JSONL"
    )
    parser.add_argument(
        "--embeddings",
        default=config.EMBEDDINGS_PATH,
        help="Output path for embeddings .npy",
    )
    parser.add_argument(
        "--index", default=config.FAISS_INDEX_PATH, help="Output path for FAISS index"
    )
    parser.add_argument(
        "--manifest",
        default=config.RETRIEVAL_MANIFEST_PATH,
        help="Output path for retrieval artifact manifest",
    )
    args = parser.parse_args()

    retriever = DenseRetriever(
        chunks_path=args.chunks,
        embeddings_path=args.embeddings,
        index_path=args.index,
        manifest_path=args.manifest,
    )

    t0 = time.time()
    embeddings = retriever.build_embeddings()
    t1 = time.time()
    print(f"Embedding took {t1 - t0:.1f}s")

    retriever.build_index(embeddings)
    t2 = time.time()
    print(f"Index build took {t2 - t1:.1f}s")
    print(f"Total: {t2 - t0:.1f}s")


if __name__ == "__main__":
    main()
