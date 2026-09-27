import sys

import pytest

from benchmarks.benchmark_live import (
    Sample,
    _http_failure_category,
    benchmark,
    main,
    summarize,
)


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


def test_empty_workload_fails_before_contacting_service(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("an invalid workload must not contact the service")

    monkeypatch.setattr("benchmarks.benchmark_live._check_ready", fail_if_called)
    with pytest.raises(ValueError, match="no benchmark questions"):
        benchmark("http://localhost:8000", [], [1], 1.0)
    with pytest.raises(ValueError, match="empty benchmark workload"):
        summarize([], 1.0)


@pytest.mark.parametrize("levels", [[], [0], [-1], [1, 0]])
def test_invalid_concurrency_fails_before_contacting_service(monkeypatch, levels):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("an invalid workload must not contact the service")

    monkeypatch.setattr("benchmarks.benchmark_live._check_ready", fail_if_called)
    with pytest.raises(ValueError, match="positive integers"):
        benchmark("http://localhost:8000", ["question"], levels, 1.0)


@pytest.mark.parametrize(
    ("contents", "extra_args", "expected_error"),
    [
        ("", [], "no benchmark questions"),
        ("question\n", ["--limit", "0"], "--limit must be a positive integer"),
    ],
)
def test_cli_rejects_empty_workloads(
    monkeypatch, tmp_path, capsys, contents, extra_args, expected_error
):
    questions_file = tmp_path / "questions.txt"
    questions_file.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_live", "--questions", str(questions_file), *extra_args],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert expected_error in capsys.readouterr().err
