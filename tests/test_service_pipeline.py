from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from llms.llm_pipeline import GenerationResult
from service.cache import SingleFlight, TTLCache
from service.pipeline import RAGPipeline


def _chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "title": "Test", "text": text}


class NotifyingSingleFlight(SingleFlight):
    def __init__(self, follower_joined: Event):
        super().__init__()
        self.follower_joined = follower_joined

    def claim(self, key):
        claim = super().claim(key)
        if not claim.leader:
            self.follower_joined.set()
        return claim


class FakeDenseRetriever:
    def __init__(self):
        self.encoded = None

    def encode_queries(self, queries):
        self.encoded = queries
        return [[1.0, 0.0]]

    def search_encoded(self, query_vectors, k):
        assert query_vectors == [[1.0, 0.0]]
        return [
            [
                _chunk("dense-1", "dense result"),
                _chunk("shared", "shared result"),
            ][:k]
        ]


class FakeSparseRetriever:
    def retrieve_top_k(self, query, k):
        return [
            _chunk("shared", "shared result"),
            _chunk("sparse-1", "sparse result"),
        ][:k]


def test_pipeline_reports_stage_timings_and_fused_chunk_ids():
    clock_values = iter(float(value) for value in range(12))
    seen = {}

    def generator(question, passages):
        seen["question"] = question
        seen["chunk_ids"] = [passage["chunk_id"] for passage in passages]
        return "Grounded answer"

    dense = FakeDenseRetriever()
    pipeline = RAGPipeline(
        dense=dense,
        sparse=FakeSparseRetriever(),
        generator=generator,
        default_top_k=3,
        clock=lambda: next(clock_values),
    )

    result = pipeline.answer("  What is RRF?  ")

    assert dense.encoded == ["What is RRF?"]
    assert result.answer == "Grounded answer"
    assert result.retrieved_chunk_ids == ["shared", "dense-1", "sparse-1"]
    assert seen == {
        "question": "What is RRF?",
        "chunk_ids": ["shared", "dense-1", "sparse-1"],
    }
    assert result.timings_seconds == {
        "embedding": 1.0,
        "dense_retrieval": 1.0,
        "bm25_retrieval": 1.0,
        "fusion": 1.0,
        "generation": 1.0,
        "total": 11.0,
    }
    assert result.fallback is False
    assert result.cache_status == "bypass"


def test_pipeline_marks_unknown_as_fallback():
    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=lambda question, passages: "Unknown",
    )

    result = pipeline.answer("Question")

    assert result.answer == "Unknown"
    assert result.fallback is True


def test_pipeline_propagates_context_safety_metadata_and_provider_short_circuit():
    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=lambda question, passages: GenerationResult(
            answer="Unknown",
            provider_called=False,
            safety_filtered_chunk_ids=("shared", "dense-1"),
        ),
    )

    result = pipeline.answer("Question")

    assert result.fallback is True
    assert result.provider_called is False
    assert result.safety_filtered_chunk_ids == ("shared", "dense-1")


def test_pipeline_cache_skips_retrieval_and_generation_on_repeat_query():
    calls = {"generation": 0}

    def generator(question, passages):
        calls["generation"] += 1
        return "Cached answer"

    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=generator,
        response_cache=TTLCache(max_entries=4, ttl_seconds=60),
    )

    first = pipeline.answer("Repeated question", use_cache=True)
    second = pipeline.answer("Repeated question", use_cache=True)
    bypassed = pipeline.answer("Repeated question", use_cache=False)

    assert first.cache_hit is False
    assert first.cache_status == "miss"
    assert first.provider_called is True
    assert second.cache_hit is True
    assert second.cache_status == "hit"
    assert second.provider_called is False
    assert second.estimated_cost_usd == 0.0
    assert set(second.timings_seconds) == {"cache_lookup", "total"}
    assert bypassed.cache_hit is False
    assert bypassed.cache_status == "bypass"
    assert calls["generation"] == 2


def test_pipeline_cache_keeps_case_sensitive_questions_distinct():
    seen_questions = []

    def generator(question, passages):
        seen_questions.append(question)
        return f"Answer for {question}"

    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=generator,
        response_cache=TTLCache(max_entries=4, ttl_seconds=60),
    )

    upper = pipeline.answer("Explain US")
    lower = pipeline.answer("Explain us")
    upper_again = pipeline.answer("  Explain US  ")

    assert upper.answer == "Answer for Explain US"
    assert lower.answer == "Answer for Explain us"
    assert upper.cache_status == lower.cache_status == "miss"
    assert upper_again.cache_status == "hit"
    assert upper_again.answer == upper.answer
    assert seen_questions == ["Explain US", "Explain us"]


def test_concurrent_cache_misses_share_one_pipeline_execution():
    entered = Event()
    release = Event()
    calls = 0
    calls_lock = Lock()
    follower_joined = Event()

    def generator(question, passages):
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return "Shared answer"

    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=generator,
        response_cache=TTLCache(max_entries=4, ttl_seconds=60),
    )
    pipeline._singleflight = NotifyingSingleFlight(follower_joined)

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(pipeline.answer, "Same question")
        assert entered.wait(timeout=1)
        follower = executor.submit(pipeline.answer, "Same question")
        assert follower_joined.wait(timeout=1)
        release.set()
        results = [leader.result(timeout=1), follower.result(timeout=1)]

    assert calls == 1
    assert {result.cache_status for result in results} == {"miss", "coalesced"}
    coalesced = next(result for result in results if result.cache_status == "coalesced")
    assert coalesced.answer == "Shared answer"
    assert coalesced.provider_called is False
    assert coalesced.cache_hit is False
    assert set(coalesced.timings_seconds) == {
        "cache_lookup",
        "coalescing_wait",
        "total",
    }


def test_failed_singleflight_propagates_and_allows_retry():
    entered = Event()
    release = Event()
    attempts = 0
    attempts_lock = Lock()
    follower_joined = Event()

    def generator(question, passages):
        nonlocal attempts
        with attempts_lock:
            attempts += 1
            attempt = attempts
        if attempt == 1:
            entered.set()
            assert release.wait(timeout=2)
            raise RuntimeError("provider failed")
        return "Recovered"

    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=generator,
        response_cache=TTLCache(max_entries=4, ttl_seconds=60),
    )
    pipeline._singleflight = NotifyingSingleFlight(follower_joined)

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(pipeline.answer, "Same question")
        assert entered.wait(timeout=1)
        follower = executor.submit(pipeline.answer, "Same question")
        assert follower_joined.wait(timeout=1)
        release.set()
        for request in (leader, follower):
            with pytest.raises(RuntimeError, match="provider failed"):
                request.result(timeout=1)

    recovered = pipeline.answer("Same question")
    assert recovered.answer == "Recovered"
    assert recovered.cache_status == "miss"
    assert attempts == 2
