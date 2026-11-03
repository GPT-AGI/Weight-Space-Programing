"""LLM, pricing, and Unix-socket host clients."""

from __future__ import annotations

import json
import os
import random
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from wsp.contracts import ModelSpec, TokenUsage


@dataclass
class LLMReply:
    content: str
    reasoning: str | None
    usage: TokenUsage
    latency_s: float


class LLMClient(Protocol):
    def complete(
        self, messages: Sequence[dict[str, str]], *, max_tokens: int
    ) -> LLMReply: ...


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("models", raw)
    if isinstance(rows, dict):
        rows = [{"id": key, **value} for key, value in rows.items()]
    specs = [ModelSpec(**row) for row in rows]
    return {spec.id: spec for spec in specs}


def usage_cost(usage: TokenUsage, spec: ModelSpec) -> float:
    cached = min(max(usage.cached, 0), max(usage.prompt, 0))
    uncached = max(0, usage.prompt - cached)
    cached_rate = (
        spec.usd_per_m_input
        if spec.usd_per_m_cached is None
        else spec.usd_per_m_cached
    )
    return (
        uncached * spec.usd_per_m_input
        + cached * cached_rate
        + max(usage.completion + usage.reasoning, 0)
        * spec.usd_per_m_output
    ) / 1_000_000


def _non_reasoning_completion(
    *,
    prompt: int,
    completion: int,
    reasoning: int,
    total: int,
) -> int:
    if reasoning <= 0:
        return completion
    if total > 0:
        return max(0, total - prompt - reasoning)
    return max(0, completion - reasoning)


def _reasoning_content(message: Any) -> str | None:
    direct = getattr(message, "reasoning_content", None)
    extra = getattr(message, "model_extra", None) or {}
    direct = direct or extra.get("reasoning_content")
    if direct:
        return str(direct)
    details = getattr(message, "reasoning_details", None)
    if details is None:
        details = extra.get("reasoning_details")
    if not isinstance(details, list):
        return None
    parts: list[str] = []
    for detail in details:
        if isinstance(detail, dict):
            text = detail.get("text")
        else:
            text = getattr(detail, "text", None)
        if text:
            parts.append(str(text))
    return "".join(parts) or None


def _load_wsp_env() -> None:
    if os.environ.get("OFOX_API_KEY"):
        return
    path = Path.home() / ".config" / "wsp" / "env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            parts = shlex.split(line, comments=True)
        except ValueError:
            continue
        if parts and parts[0] == "export":
            parts = parts[1:]
        for assignment in parts:
            key, separator, value = assignment.partition("=")
            if separator and key in {
                "OFOX_API_KEY",
                "OFOX_BASE_URL",
                "WSP_AGENT_MODEL",
            }:
                os.environ.setdefault(key, value)


class OfoxClient:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        timeout_s: float = 180.0,
        max_retries: int = 5,
    ) -> None:
        _load_wsp_env()
        key = os.environ.get("OFOX_API_KEY")
        if not key:
            raise RuntimeError(
                "OFOX_API_KEY is unset and ~/.config/wsp/env did not provide it"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install the 'agent' extra to use OfoxAI") from exc
        self.spec = spec
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=key,
            base_url=os.environ.get("OFOX_BASE_URL", "https://api.ofox.ai/v1"),
            timeout=timeout_s,
            max_retries=0,
        )

    def complete(
        self, messages: Sequence[dict[str, str]], *, max_tokens: int
    ) -> LLMReply:
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.spec.id,
                    "messages": list(messages),
                    "max_tokens": max_tokens,
                }
                if self.spec.extra_body:
                    kwargs["extra_body"] = self.spec.extra_body
                response = self._client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                usage_obj = response.usage
                prompt_details = getattr(
                    usage_obj, "prompt_tokens_details", None
                )
                completion_details = getattr(
                    usage_obj, "completion_tokens_details", None
                )
                prompt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
                raw_completion = int(
                    getattr(usage_obj, "completion_tokens", 0) or 0
                )
                reasoning = int(
                    getattr(completion_details, "reasoning_tokens", 0) or 0
                )
                total = int(getattr(usage_obj, "total_tokens", 0) or 0)
                usage = TokenUsage(
                    prompt=prompt,
                    completion=_non_reasoning_completion(
                        prompt=prompt,
                        completion=raw_completion,
                        reasoning=reasoning,
                        total=total,
                    ),
                    cached=int(
                        getattr(prompt_details, "cached_tokens", 0) or 0
                    ),
                    reasoning=reasoning,
                )
                return LLMReply(
                    content=message.content or "",
                    reasoning=_reasoning_content(message),
                    usage=usage,
                    latency_s=time.monotonic() - started,
                )
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or (
                    isinstance(status, int) and 500 <= status < 600
                )
                retryable = retryable or exc.__class__.__name__ in {
                    "APITimeoutError",
                    "APIConnectionError",
                    "RateLimitError",
                }
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = min(30.0, 1.0 * (2**attempt))
                time.sleep(delay + random.random() * min(1.0, delay / 4))
        raise RuntimeError(f"OfoxAI request failed: {last_error}")


class FakeLLMClient:
    def __init__(self, replies: Sequence[LLMReply]) -> None:
        self._replies = list(replies)
        self._position = 0

    @classmethod
    def from_file(cls, path: Path) -> FakeLLMClient:
        text = path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            rows = parsed.get("replies", []) if isinstance(parsed, dict) else parsed
        except json.JSONDecodeError:
            rows = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        if not isinstance(rows, list):
            raise TypeError("fake LLM file must contain a JSON list or JSONL")
        replies: list[LLMReply] = []
        for row in rows:
            if isinstance(row, str):
                row = {"content": row}
            usage = row.get("usage", {})
            replies.append(
                LLMReply(
                    content=str(row.get("content", "")),
                    reasoning=row.get("reasoning_content", row.get("reasoning")),
                    usage=TokenUsage(**usage),
                    latency_s=float(row.get("latency_s", 0.0)),
                )
            )
        return cls(replies)

    def complete(
        self, messages: Sequence[dict[str, str]], *, max_tokens: int
    ) -> LLMReply:
        del messages, max_tokens
        if self._position >= len(self._replies):
            raise RuntimeError("fake LLM reply list was exhausted")
        reply = self._replies[self._position]
        self._position += 1
        return reply


class UnixHostClient:
    def __init__(
        self,
        socket_path: Path,
        *,
        session: str | None = None,
        token: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.session = session
        self.token = token
        transport = httpx.HTTPTransport(uds=str(socket_path))
        self._client = httpx.Client(
            transport=transport,
            base_url="http://wsp",
            timeout=timeout_s,
        )

    def post(self, endpoint: str, **payload: Any) -> dict[str, Any]:
        if self.session is not None:
            payload.setdefault("session", self.session)
        headers = (
            {"Authorization": f"Bearer {self.token}"}
            if self.token is not None
            else {}
        )
        response = self._client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError(f"{endpoint} returned a non-object response")
        return value

    def close(self) -> None:
        self._client.close()
