from benchmarks.benchmark_live import Sample, _http_failure_category, summarize


def test_summary_reports_latency_usage_cost_cache_and_failures():
    samples = [
        Sample(
            elapsed_seconds=0.1,
            payload={
                "fallback": False,
                "cache_hit": False,
                "provider_called": True,
                "ttft_ms": 40.0,
                "completion_tokens": 10,
                "estimated_cost_usd": 0.002,
                "timings_ms": {"generation": 80.0},
            },
        ),
        Sample(
            elapsed_seconds=0.02,
            payload={
                "fallback": False,
                "cache_hit": True,
                "provider_called": False,
                "ttft_ms": 0.0,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.0,
                "timings_ms": {"total": 10.0},
            },
        ),
        Sample(elapsed_seconds=1.0, payload=None, failure="timeout"),
    ]

    result = summarize(samples, wall_seconds=1.5)

    assert result["successes"] == 2
    assert result["failure_rate"] == 0.3333
    assert result["failure_categories"] == {"timeout": 1}
    assert result["cache_hit_rate"] == 0.5
    assert result["ttft_ms"] == {"p50": 40.0, "p95": 40.0}
    assert result["completion_tokens_per_second"] == 125.0
    assert result["provider_cost_usd_total"] == 0.002
    assert result["provider_cost_usd_per_question"] == 0.001


def test_http_502_is_reported_as_provider_failure():
    assert _http_failure_category(502) == "provider"
    assert _http_failure_category(500) == "http_500"
