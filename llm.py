import json
import os
import time
from dataclasses import dataclass

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALLOWED_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "qwen/qwen3-8b",
    "qwen/qwen-2.5-7b-instruct",
    "allenai/olmo-3-7b-instruct",
    "mistralai/mistral-7b-instruct",
]
DEFAULT_MODEL = ALLOWED_MODELS[0]


@dataclass(frozen=True)
class LLMResponse:
    """Text plus provider telemetry from one streamed completion."""

    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    ttft_seconds: float | None


def call_llm(
    query: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: int = 30,
) -> str:
    """
    Call OpenRouter chat completions and return the assistant text.

    Constraints:
    - Uses OPENROUTER_API_KEY from environment.
    - Allows only models in ALLOWED_MODELS.
    """
    assert model in ALLOWED_MODELS, (
        f"Model '{model}' is not allowed. Allowed models: {ALLOWED_MODELS}"
    )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        raise RuntimeError("OpenRouter request timed out") from None
    except requests.ConnectionError as e:
        raise RuntimeError(f"OpenRouter connection failed: {e}") from None
    except requests.HTTPError as e:
        raise RuntimeError(f"OpenRouter HTTP error: {e}") from None
    except ValueError as e:
        raise RuntimeError(f"OpenRouter invalid JSON: {e}") from None

    if data.get("choices"):
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"OpenRouter response missing expected content: {data}")

    raise RuntimeError(f"OpenRouter response missing choices: {data}")


def call_llm_with_metrics(
    query: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: int = 30,
) -> LLMResponse:
    """Stream a completion and return TTFT, token usage, and provider cost."""
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"Model '{model}' is not allowed. Allowed models: {ALLOWED_MODELS}"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    started = time.perf_counter()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()

        content_parts = []
        usage: dict[str, object] = {}
        ttft_seconds = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or raw_line.startswith(":"):
                continue
            line = raw_line[5:].strip() if raw_line.startswith("data:") else raw_line
            if line == "[DONE]":
                break
            payload = json.loads(line)
            if payload.get("usage"):
                usage = payload["usage"]
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            token = delta.get("content")
            if token:
                if ttft_seconds is None:
                    ttft_seconds = time.perf_counter() - started
                content_parts.append(token)
    except requests.Timeout:
        raise RuntimeError("OpenRouter request timed out") from None
    except requests.ConnectionError as exc:
        raise RuntimeError(f"OpenRouter connection failed: {exc}") from None
    except requests.HTTPError as exc:
        raise RuntimeError(f"OpenRouter HTTP error: {exc}") from None
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"OpenRouter invalid streaming response: {exc}") from None

    content = "".join(content_parts).strip()
    if not content:
        raise RuntimeError("OpenRouter streaming response contained no text")

    def optional_int(name: str) -> int | None:
        value = usage.get(name)
        return int(value) if value is not None else None

    cost = usage.get("cost")
    return LLMResponse(
        content=content,
        prompt_tokens=optional_int("prompt_tokens"),
        completion_tokens=optional_int("completion_tokens"),
        total_tokens=optional_int("total_tokens"),
        cost_usd=float(cost) if cost is not None else None,
        ttft_seconds=ttft_seconds,
    )
