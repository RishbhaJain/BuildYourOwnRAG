import json
import numpy as np
from rank_bm25 import BM25Okapi

import config


class BM25Retriever:
    def __init__(self, chunks_path=config.CHUNKS_JSONL_PATH):
        self.chunks_path = chunks_path
        self.bm25 = None
        self.chunks = []

    def load_bm25(self):
        self.chunks = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        tokenized_chunks = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve_top_k(self, query, k=config.BM25_TOP_K):
        if self.bm25 is None:
            self.load_bm25()

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [dict(self.chunks[i], bm25_score=float(scores[i])) for i in top_k_idx]