"""FastAPI application for online hybrid RAG inference."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from llms.llm_pipeline import ProviderGenerationError
from service.metrics import ServiceMetrics
from service.pipeline import RAGPipeline, create_pipeline_from_environment

logger = logging.getLogger(__name__)
PipelineFactory = Callable[[], RAGPipeline]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class AnswerResponse(BaseModel):
    answer: str
    retrieved_chunk_ids: list[str]
    fallback: bool
    timings_ms: dict[str, float]
    cache_hit: bool
    cache_status: Literal["hit", "miss", "bypass"]
    provider_called: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    ttft_ms: float | None


def create_app(
    pipeline_factory: PipelineFactory = create_pipeline_from_environment,
) -> FastAPI:
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        app.state.startup_error = None
        try:
            app.state.pipeline = await run_in_threadpool(pipeline_factory)
            metrics.set_ready(True)
            logger.info("RAG pipeline loaded and ready.")
        except Exception as exc:
            app.state.startup_error = type(exc).__name__
            metrics.set_ready(False)
            logger.exception("RAG pipeline failed to initialize.")
        yield
        metrics.set_ready(False)

    app = FastAPI(
        title="BuildYourOwnRAG API",
        version="1.0.0",
        description="Measured hybrid retrieval and generation service.",
        lifespan=lifespan,
    )
    app.state.metrics = metrics

    @app.get("/")
    async def root():
        return {
            "service": "BuildYourOwnRAG",
            "docs": "/docs",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if app.state.pipeline is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": app.state.startup_error or "pipeline_not_loaded",
                },
            )
        return {"status": "ready"}

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(request: AnswerRequest):
        pipeline = app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline is not ready")

        started = time.perf_counter()
        try:
            result = await run_in_threadpool(
                pipeline.answer,
                request.question,
                request.top_k,
                request.use_cache,
            )
        except ProviderGenerationError:
            metrics.observe_failure(
                time.perf_counter() - started,
                category="provider",
            )
            logger.exception("LLM provider failed during RAG inference.")
            raise HTTPException(status_code=502, detail="LLM provider failed") from None
        except Exception:
            metrics.observe_failure(
                time.perf_counter() - started,
                category="internal",
            )
            logger.exception("RAG answer request failed.")
            raise HTTPException(
                status_code=500, detail="RAG inference failed"
            ) from None

        metrics.observe_success(
            result.timings_seconds,
            result.fallback,
            time.perf_counter() - started,
            cache_result=result.cache_status,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
        )
        return AnswerResponse(
            answer=result.answer,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
            fallback=result.fallback,
            timings_ms={
                stage: round(seconds * 1_000, 3)
                for stage, seconds in result.timings_seconds.items()
            },
            cache_hit=result.cache_hit,
            cache_status=result.cache_status,
            provider_called=result.provider_called,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            ttft_ms=(
                round(result.ttft_seconds * 1_000, 3)
                if result.ttft_seconds is not None
                else None
            ),
        )

    @app.get("/metrics")
    async def prometheus_metrics():
        payload, content_type = metrics.render()
        return Response(payload, headers={"Content-Type": content_type})

    return app


app = create_app()
