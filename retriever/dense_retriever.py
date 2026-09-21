"""
Dense retriever using FAISS for nearest-neighbor search over embeddings.

Builds or loads a FAISS index from pre-computed chunk embeddings.
At query time, encodes the question and returns top-k chunks by cosine similarity.
"""

import json
import logging
import os

import faiss
import numpy as np

import config
from embedder.embedder import Embedder

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(
        self,
        chunks_path: str = config.CHUNKS_JSONL_PATH,
        embeddings_path: str = config.EMBEDDINGS_PATH,
        index_path: str = config.FAISS_INDEX_PATH,
    ):
        self.chunks_path = chunks_path
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.chunks = []
        self.index = None
        self.embedder = Embedder()

    def load_chunks(self):
        """Load chunk metadata from JSONL."""
        self.chunks = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        logger.info("Loaded %d chunks from %s", len(self.chunks), self.chunks_path)

    def build_embeddings(self):
        """Embed all chunks and save the numpy array to disk."""
        if not self.chunks:
            self.load_chunks()

        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedder.encode_passages(texts)

        os.makedirs(os.path.dirname(self.embeddings_path) or ".", exist_ok=True)
        np.save(self.embeddings_path, embeddings)
        logger.info(
            "Saved embeddings (%s) to %s", embeddings.shape, self.embeddings_path
        )
        return embeddings

    def build_index(self, embeddings: np.ndarray = None):
        """Build a FAISS index from embeddings and save to disk."""
        if embeddings is None:
            if os.path.exists(self.embeddings_path):
                embeddings = np.load(self.embeddings_path)
                logger.info("Loaded embeddings from %s", self.embeddings_path)
            else:
                embeddings = self.build_embeddings()

        # Inner product index (embeddings are L2-normalized, so IP == cosine similarity)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        logger.info(
            "Built and saved FAISS index (%d vectors, dim=%d) to %s",
            self.index.ntotal,
            dim,
            self.index_path,
        )

    def load_index(self):
        """Load a pre-built FAISS index and chunk metadata."""
        if not self.chunks:
            self.load_chunks()
        self.index = faiss.read_index(self.index_path)
        logger.info(
            "Loaded FAISS index (%d vectors) from %s",
            self.index.ntotal,
            self.index_path,
        )

    def retrieve_top_k(self, query: str, k: int = config.DENSE_TOP_K) -> list[dict]:
        """Retrieve top-k chunks for a single query."""
        return self.batch_retrieve_top_k([query], k=k)[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries separately so callers can measure embedding latency."""
        return self.embedder.encode_queries(queries)

    def search_encoded(
        self,
        query_vectors: np.ndarray,
        k: int = config.DENSE_TOP_K,
    ) -> list[list[dict]]:
        """Search the FAISS index with pre-encoded query vectors.

        Separating encoding from search keeps the batch API unchanged while
        allowing online serving code to report embedding and index-search
        latency independently.
        """
        if self.index is None:
            self.load_index()

        scores_batch, indices_batch = self.index.search(
            query_vectors.astype(np.float32), k
        )

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                results.append(dict(self.chunks[idx], dense_score=float(score)))
            all_results.append(results)
        return all_results

    def batch_retrieve_top_k(
        self, queries: list[str], k: int = config.DENSE_TOP_K
    ) -> list[list[dict]]:
        """Retrieve top-k chunks for a batch of queries at once."""
        query_vectors = self.encode_queries(queries)
        return self.search_encoded(query_vectors, k=k)
