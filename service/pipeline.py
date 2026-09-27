"""Reusable online RAG pipeline with per-stage timing."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from retriever.fusion import reciprocal_rank_fusion
from service.cache import TTLCache

logger = logging.getLogger(__name__)


class DenseRetrieverLike(Protocol):
    def encode_queries(self, queries: list[str]) -> Any: ...

    def search_encoded(self, query_vectors: Any, k: int) -> list[list[dict]]: ...


class SparseRetrieverLike(Protocol):
    def retrieve_top_k(self, query: str, k: int) -> list[dict]: ...


Generator = Callable[[str, list[dict]], str]
Clock = Callable[[], float]
CacheStatus = Literal["hit", "miss", "bypass"]


@dataclass(frozen=True)
class RAGResult:
    """Structured result returned by the serving pipeline."""

    answer: str
    retrieved_chunk_ids: list[str]
    timings_seconds: dict[str, float]
    fallback: bool
    cache_hit: bool = False
    cache_status: CacheStatus = "bypass"
    provider_called: bool = True
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    ttft_seconds: float | None = None


class RAGPipeline:
    """Hybrid retrieval and generation pipeline loaded once per process."""

    def __init__(
        self,
        dense: DenseRetrieverLike,
        sparse: SparseRetrieverLike,
        generator: Generator,
        *,
        default_top_k: int = 5,
        clock: Clock = time.perf_counter,
        response_cache: TTLCache[RAGResult] | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.generator = generator
        self.default_top_k = default_top_k
        self.clock = clock
        self.response_cache = response_cache

    def _measure(self, timings: dict[str, float], stage: str, operation):
        started = self.clock()
        value = operation()
        timings[stage] = self.clock() - started
        return value

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> RAGResult:
        """Answer one question and expose latency for every major stage."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        selected_top_k = top_k or self.default_top_k
        if selected_top_k < 1:
            raise ValueError("top_k must be at least 1")

        total_started = self.clock()
        timings: dict[str, float] = {}

        cache_key = f"{selected_top_k}\0{question.casefold()}"
        if use_cache and self.response_cache is not None:
            cache_started = self.clock()
            cached = self.response_cache.get(cache_key)
            timings["cache_lookup"] = self.clock() - cache_started
            if cached is not None:
                timings["total"] = self.clock() - total_started
                return RAGResult(
                    answer=cached.answer,
                    retrieved_chunk_ids=cached.retrieved_chunk_ids,
                    timings_seconds=timings,
                    fallback=cached.fallback,
                    cache_hit=True,
                    cache_status="hit",
                    provider_called=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    ttft_seconds=0.0,
                )

        query_vectors = self._measure(
            timings,
            "embedding",
            lambda: self.dense.encode_queries([question]),
        )
        dense_results = self._measure(
            timings,
            "dense_retrieval",
            lambda: self.dense.search_encoded(query_vectors, k=selected_top_k)[0],
        )
        sparse_results = self._measure(
            timings,
            "bm25_retrieval",
            lambda: self.sparse.retrieve_top_k(question, k=selected_top_k),
        )
        fused_results = self._measure(
            timings,
            "fusion",
            lambda: reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                top_k=selected_top_k,
            ),
        )
        generated = self._measure(
            timings,
            "generation",
            lambda: self.generator(question, fused_results),
        )
        timings["total"] = self.clock() - total_started

        if isinstance(generated, str):
            answer = generated
            metadata = {}
        else:
            answer = generated.answer
            metadata = {
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
                "total_tokens": generated.total_tokens,
                "estimated_cost_usd": generated.cost_usd,
                "ttft_seconds": generated.ttft_seconds,
            }
        normalized_answer = answer.strip()
        result = RAGResult(
            answer=normalized_answer or "Unknown",
            retrieved_chunk_ids=[
                str(chunk["chunk_id"])
                for chunk in fused_results
                if chunk.get("chunk_id") is not None
            ],
            timings_seconds=timings,
            fallback=(not normalized_answer or normalized_answer.lower() == "unknown"),
            cache_status=(
                "miss" if use_cache and self.response_cache is not None else "bypass"
            ),
            **metadata,
        )
        if use_cache and self.response_cache is not None and not result.fallback:
            self.response_cache.set(cache_key, result)
        return result


def load_production_pipeline() -> RAGPipeline:
    """Load model, FAISS index, corpus, and BM25 state once at startup."""
    import config
    from llms.llm_pipeline import generate_answer_with_metrics
    from retriever.bm25_retriever import BM25Retriever
    from retriever.dense_retriever import DenseRetriever

    model_path = Path(config.EMBEDDING_MODEL)
    auto_download = os.getenv("RAG_AUTO_DOWNLOAD_MODEL", "0") == "1"
    if not model_path.exists() and auto_download:
        logger.info("Embedding model is missing; downloading it before startup.")
        from download_model import main as download_model

        download_model()

    dense = DenseRetriever()
    dense.load_chunks()
    if Path(dense.index_path).exists():
        dense.load_index()
    else:
        logger.info("FAISS index is missing; rebuilding it from stored embeddings.")
        dense.build_index()
    dense.embedder.load_model()

    sparse = BM25Retriever()
    sparse.load_bm25()

    cache_entries = int(os.getenv("RAG_CACHE_MAX_ENTRIES", "256"))
    cache_ttl = float(os.getenv("RAG_CACHE_TTL_SECONDS", "300"))
    response_cache = (
        TTLCache[RAGResult](cache_entries, cache_ttl) if cache_entries > 0 else None
    )

    return RAGPipeline(
        dense=dense,
        sparse=sparse,
        generator=generate_answer_with_metrics,
        default_top_k=config.DENSE_TOP_K,
        response_cache=response_cache,
    )


class _MockDenseRetriever:
    def encode_queries(self, queries: list[str]) -> list[str]:
        return queries

    def search_encoded(self, query_vectors: list[str], k: int) -> list[list[dict]]:
        return [
            [
                {
                    "chunk_id": "mock-dense-1",
                    "title": "Berkeley EECS",
                    "text": "The mock service is ready for deterministic integration tests.",
                }
            ][:k]
        ]


class _MockSparseRetriever:
    def retrieve_top_k(self, query: str, k: int) -> list[dict]:
        return [
            {
                "chunk_id": "mock-sparse-1",
                "title": "Service test",
                "text": "Mock retrieval avoids model downloads and external API calls.",
            }
        ][:k]


def build_mock_pipeline() -> RAGPipeline:
    """Build a credential-free pipeline for tests and container smoke checks."""

    def mock_generator(question: str, passages: list[dict]) -> str:
        return "Mock answer"

    return RAGPipeline(
        dense=_MockDenseRetriever(),
        sparse=_MockSparseRetriever(),
        generator=mock_generator,
        default_top_k=5,
        response_cache=TTLCache[RAGResult](max_entries=32, ttl_seconds=60),
    )


def create_pipeline_from_environment() -> RAGPipeline:
    """Select a production or deterministic mock pipeline from RAG_MODE."""
    mode = os.getenv("RAG_MODE", "production").strip().lower()
    if mode == "mock":
        logger.warning("Starting the RAG API in mock mode.")
        return build_mock_pipeline()
    if mode != "production":
        raise ValueError("RAG_MODE must be either 'production' or 'mock'")
    return load_production_pipeline()
