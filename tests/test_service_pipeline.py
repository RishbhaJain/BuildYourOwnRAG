from service.pipeline import RAGPipeline


def _chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "title": "Test", "text": text}


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


def test_pipeline_marks_unknown_as_fallback():
    pipeline = RAGPipeline(
        dense=FakeDenseRetriever(),
        sparse=FakeSparseRetriever(),
        generator=lambda question, passages: "Unknown",
    )

    result = pipeline.answer("Question")

    assert result.answer == "Unknown"
    assert result.fallback is True
