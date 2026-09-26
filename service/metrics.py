"""Prometheus instrumentation for the online RAG service."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class ServiceMetrics:
    """Own a per-application registry to keep tests and workers isolated."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "rag_requests_total",
            "RAG answer requests by outcome.",
            ("status",),
            registry=self.registry,
        )
        self.failures = Counter(
            "rag_request_failures_total",
            "RAG answer requests that raised an exception by failure category.",
            ("category",),
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "rag_fallbacks_total",
            "RAG answers that returned the fallback value.",
            registry=self.registry,
        )
        self.cache_requests = Counter(
            "rag_cache_requests_total",
            "RAG requests by cache outcome.",
            ("result",),
            registry=self.registry,
        )
        self.provider_tokens = Counter(
            "rag_provider_tokens_total",
            "Provider tokens consumed by type.",
            ("type",),
            registry=self.registry,
        )
        self.provider_cost = Counter(
            "rag_provider_cost_usd_total",
            "Provider-reported generation cost in US dollars.",
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "rag_request_latency_seconds",
            "End-to-end latency for answer requests.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )
        self.stage_latency = Histogram(
            "rag_stage_latency_seconds",
            "Latency for individual RAG pipeline stages.",
            ("stage",),
            buckets=(
                0.001,
                0.0025,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2.5,
                5,
                10,
            ),
            registry=self.registry,
        )
        self.ready = Gauge(
            "rag_service_ready",
            "Whether the production pipeline completed startup.",
            registry=self.registry,
        )
        self.ready.set(0)

    def set_ready(self, is_ready: bool) -> None:
        self.ready.set(1 if is_ready else 0)

    def observe_success(
        self,
        timings: dict[str, float],
        fallback: bool,
        request_elapsed_seconds: float,
        cache_result: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        estimated_cost_usd: float | None,
    ) -> None:
        self.requests.labels(status="success").inc()
        self.request_latency.observe(request_elapsed_seconds)
        for stage, seconds in timings.items():
            if stage != "total":
                self.stage_latency.labels(stage=stage).observe(seconds)
        if fallback:
            self.fallbacks.inc()
        self.cache_requests.labels(result=cache_result).inc()
        if prompt_tokens:
            self.provider_tokens.labels(type="prompt").inc(prompt_tokens)
        if completion_tokens:
            self.provider_tokens.labels(type="completion").inc(completion_tokens)
        if total_tokens:
            self.provider_tokens.labels(type="total").inc(total_tokens)
        if estimated_cost_usd:
            self.provider_cost.inc(estimated_cost_usd)

    def observe_failure(self, elapsed_seconds: float, category: str) -> None:
        self.requests.labels(status="error").inc()
        self.failures.labels(category=category).inc()
        self.request_latency.observe(elapsed_seconds)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
