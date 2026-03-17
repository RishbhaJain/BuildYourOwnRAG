"""
Dense embedder using sentence-transformers.

Encodes corpus chunks and queries into vector embeddings for FAISS retrieval.
BGE models use an instruction prefix for queries to improve retrieval quality.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        query_prefix: str = config.EMBEDDING_QUERY_PREFIX,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model = None

    def load_model(self):
        """Load the sentence-transformer model (lazy, so import cost is deferred)."""
        if self.model is not None:
            return
        logger.info("Loading embedding model: %s", self.model_name)
        import torch
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        # Use MPS on Mac for local dev, but Gradescope has no GPU so it'll be CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        self.model = SentenceTransformer(self.model_name, device=device, local_files_only=True)
        logger.info("Model loaded — embedding dim: %d", self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode corpus passages (no prefix)."""
        self.load_model()
        logger.info("Encoding %d passages (batch_size=%d)", len(texts), self.batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode queries with the BGE instruction prefix."""
        self.load_model()
        prefixed = [self.query_prefix + q for q in queries]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings
