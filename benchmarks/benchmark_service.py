"""Measure credential-free FastAPI serving overhead and write JSON/Markdown."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from fastapi.testclient import TestClient

from service.app import create_app
from service.pipeline import build_mock_pipeline


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def run_benchmark(iterations: int, warmup: int) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    app = create_app(build_mock_pipeline)
    request = {"question": "What is the measured serving overhead?", "top_k": 2}
    latencies_ms: list[float] = []

    with TestClient(app) as client:
        for _ in range(warmup):
            response = client.post("/answer", json=request)
            response.raise_for_status()

        run_started = time.perf_counter()
        for _ in range(iterations):
            started = time.perf_counter()
            response = client.post("/answer", json=request)
            response.raise_for_status()
            latencies_ms.append((time.perf_counter() - started) * 1_000)
        elapsed = time.perf_counter() - run_started

    return {
        "configuration": "FastAPI TestClient with deterministic mocked retrieval and generation",
        "iterations": iterations,
        "warmup_requests": warmup,
        "p50_latency_ms": round(percentile(latencies_ms, 0.50), 3),
        "p95_latency_ms": round(percentile(latencies_ms, 0.95), 3),
        "throughput_requests_per_second": round(iterations / elapsed, 2),
    }


def render_markdown(result: dict) -> str:
    return (
        "# Mock service benchmark\n\n"
        "This benchmark isolates API and orchestration overhead. It does not "
        "represent live model or OpenRouter latency.\n\n"
        "| Configuration | Requests | p50 latency | p95 latency | Throughput |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Deterministic mock | {result['iterations']} | "
        f"{result['p50_latency_ms']:.3f} ms | "
        f"{result['p95_latency_ms']:.3f} ms | "
        f"{result['throughput_requests_per_second']:.2f} req/s |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
    )
    args = parser.parse_args()

    result = run_benchmark(args.iterations, args.warmup)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mock_service.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "mock_service.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
