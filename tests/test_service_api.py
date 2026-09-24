from fastapi.testclient import TestClient

from llms.llm_pipeline import ProviderGenerationError
from service.app import create_app
from service.pipeline import build_mock_pipeline


def test_health_readiness_answer_and_metrics():
    with TestClient(create_app(build_mock_pipeline)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}

        response = client.post(
            "/answer",
            json={"question": "What does the service return?", "top_k": 2},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == "Mock answer"
        assert payload["fallback"] is False
        assert payload["cache_hit"] is False
        assert payload["provider_called"] is True
        assert payload["retrieved_chunk_ids"] == [
            "mock-dense-1",
            "mock-sparse-1",
        ]
        assert set(payload["timings_ms"]) == {
            "embedding",
            "dense_retrieval",
            "bm25_retrieval",
            "fusion",
            "generation",
            "cache_lookup",
            "total",
        }

        cached = client.post(
            "/answer",
            json={"question": "What does the service return?", "top_k": 2},
        ).json()
        assert cached["cache_hit"] is True
        assert cached["provider_called"] is False
        assert cached["estimated_cost_usd"] == 0.0
        assert set(cached["timings_ms"]) == {"cache_lookup", "total"}

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert 'rag_requests_total{status="success"} 2.0' in metrics.text
        assert 'rag_stage_latency_seconds_count{stage="embedding"} 1.0' in metrics.text
        assert 'rag_cache_requests_total{result="miss"} 1.0' in metrics.text
        assert 'rag_cache_requests_total{result="hit"} 1.0' in metrics.text
        assert "rag_service_ready 1.0" in metrics.text


def test_whitespace_question_is_rejected():
    with TestClient(create_app(build_mock_pipeline)) as client:
        response = client.post("/answer", json={"question": "   "})
        assert response.status_code == 422


def test_failed_startup_keeps_liveness_endpoint_available():
    def fail_to_load():
        raise RuntimeError("model unavailable")

    with TestClient(create_app(fail_to_load)) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "reason": "RuntimeError",
        }


def test_inference_failure_is_counted_without_leaking_exception_details():
    class BrokenPipeline:
        def answer(self, question, top_k, use_cache):
            raise RuntimeError("secret backend detail")

    with TestClient(create_app(lambda: BrokenPipeline())) as client:
        response = client.post("/answer", json={"question": "test"})
        assert response.status_code == 500
        assert response.json() == {"detail": "RAG inference failed"}
        metrics = client.get("/metrics").text
        assert 'rag_request_failures_total{category="internal"} 1.0' in metrics


def test_provider_failure_returns_bad_gateway_and_is_counted_separately():
    class ProviderFailurePipeline:
        def answer(self, question, top_k, use_cache):
            raise ProviderGenerationError("private provider detail")

    with TestClient(create_app(lambda: ProviderFailurePipeline())) as client:
        response = client.post("/answer", json={"question": "test"})
        assert response.status_code == 502
        assert response.json() == {"detail": "LLM provider failed"}
        metrics = client.get("/metrics").text
        assert 'rag_requests_total{status="error"} 1.0' in metrics
        assert 'rag_request_failures_total{category="provider"} 1.0' in metrics
