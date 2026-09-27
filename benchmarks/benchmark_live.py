"""Benchmark a running RAG API with live provider telemetry and cache comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    elapsed_seconds: float
    payload: dict[str, object] | None
    failure: str | None = None


def _http_failure_category(status_code: int) -> str:
    return "provider" if status_code == 502 else f"http_{status_code}"


def _request_answer(
    base_url: str,
    question: str,
    *,
    use_cache: bool,
    timeout_seconds: float,
) -> Sample:
    body = json.dumps(
        {"question": question, "top_k": 5, "use_cache": use_cache}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/answer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return Sample(
            time.perf_counter() - started,
            None,
            _http_failure_category(exc.code),
        )
    except TimeoutError:
        return Sample(time.perf_counter() - started, None, "timeout")
    except urllib.error.URLError as exc:
        category = "timeout" if isinstance(exc.reason, TimeoutError) else "connection"
        return Sample(time.perf_counter() - started, None, category)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return Sample(time.perf_counter() - started, None, "invalid_json")
    return Sample(time.perf_counter() - started, payload)


def run_batch(
    base_url: str,
    questions: list[str],
    *,
    concurrency: int,
    use_cache: bool,
    timeout_seconds: float,
) -> tuple[list[Sample], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(
            executor.map(
                lambda question: _request_answer(
                    base_url,
                    question,
                    use_cache=use_cache,
                    timeout_seconds=timeout_seconds,
                ),
                questions,
            )
        )
    return samples, time.perf_counter() - started


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(samples: list[Sample], wall_seconds: float) -> dict[str, object]:
    successes = [sample for sample in samples if sample.payload is not None]
    latencies_ms = [sample.elapsed_seconds * 1_000 for sample in successes]
    ttft_ms = [
        float(sample.payload["ttft_ms"])
        for sample in successes
        if sample.payload.get("provider_called")
        and sample.payload.get("ttft_ms") is not None
    ]
    completion_tokens = sum(
        int(sample.payload.get("completion_tokens") or 0) for sample in successes
    )
    generation_seconds = sum(
        float(sample.payload.get("timings_ms", {}).get("generation", 0)) / 1_000
        for sample in successes
    )
    reported_costs = [
        float(sample.payload["estimated_cost_usd"])
        for sample in successes
        if sample.payload.get("estimated_cost_usd") is not None
    ]
    failures = Counter(sample.failure for sample in samples if sample.failure)
    fallback_count = sum(bool(sample.payload.get("fallback")) for sample in successes)
    cache_hits = sum(bool(sample.payload.get("cache_hit")) for sample in successes)

    return {
        "requests": len(samples),
        "successes": len(successes),
        "failure_rate": round((len(samples) - len(successes)) / len(samples), 4),
        "failure_categories": dict(sorted(failures.items())),
        "fallback_rate": round(fallback_count / len(successes), 4)
        if successes
        else None,
        "cache_hit_rate": round(cache_hits / len(successes), 4) if successes else None,
        "latency_ms": {
            "p50": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
            "p95": round(_percentile(latencies_ms, 0.95), 3) if latencies_ms else None,
        },
        "ttft_ms": {
            "p50": round(statistics.median(ttft_ms), 3) if ttft_ms else None,
            "p95": round(_percentile(ttft_ms, 0.95), 3) if ttft_ms else None,
        },
        "throughput_requests_per_second": round(len(samples) / wall_seconds, 3),
        "completion_tokens_per_second": (
            round(completion_tokens / generation_seconds, 3)
            if generation_seconds > 0
            else None
        ),
        "provider_cost_usd_total": (
            round(sum(reported_costs), 8) if reported_costs else None
        ),
        "provider_cost_usd_per_question": (
            round(sum(reported_costs) / len(successes), 8)
            if reported_costs and successes
            else None
        ),
    }


def _check_ready(base_url: str, timeout_seconds: float) -> None:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/ready", timeout=timeout_seconds
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "ready":
        raise RuntimeError(f"service is not ready: {payload}")


def benchmark(
    base_url: str,
    questions: list[str],
    concurrency_levels: list[int],
    timeout_seconds: float,
) -> dict[str, object]:
    _check_ready(base_url, timeout_seconds)
    results = []
    for concurrency in concurrency_levels:
        samples, wall = run_batch(
            base_url,
            questions,
            concurrency=concurrency,
            use_cache=False,
            timeout_seconds=timeout_seconds,
        )
        results.append(
            {
                "mode": "baseline_cache_bypassed",
                "concurrency": concurrency,
                **summarize(samples, wall),
            }
        )

    warmup_samples, warmup_wall = run_batch(
        base_url,
        questions,
        concurrency=1,
        use_cache=True,
        timeout_seconds=timeout_seconds,
    )
    for concurrency in concurrency_levels:
        samples, wall = run_batch(
            base_url,
            questions,
            concurrency=concurrency,
            use_cache=True,
            timeout_seconds=timeout_seconds,
        )
        results.append(
            {
                "mode": "optimized_warm_cache",
                "concurrency": concurrency,
                **summarize(samples, wall),
            }
        )

    return {
        "schema_version": 1,
        "endpoint": base_url,
        "question_count": len(questions),
        "concurrency_levels": concurrency_levels,
        "cache_warmup": summarize(warmup_samples, warmup_wall),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=Path, default=Path("questions.txt"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    questions = [
        line.strip()
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    result = benchmark(
        args.base_url,
        questions,
        concurrency_levels,
        args.timeout,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
