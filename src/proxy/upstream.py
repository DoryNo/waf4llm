from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any

import httpx

from src.config.settings import Settings
from src.observability.logging import get_logger
from src.observability.metrics import metrics

logger = get_logger("upstream")


class UpstreamClient:
    INTERNAL_BODY_KEYS = {
        "rag_chunks",
        "rag_context",
        "retrieved_documents",
        "retrieved_chunks",
        "documents",
        "context",
        "sources",
        "knowledge_base",
        "search_results",
        "tool_outputs",
        "tool_results",
        "function_outputs",
        "tool_calls_output",
    }
    INTERNAL_MESSAGE_KEYS = {
        "provenance",
        "origin",
        "_provenance",
        "data_provenance",
        "source",
        "provider",
        "channel",
        "retrieval_source",
        "metadata",
        "_waf_segment_index",
        "_waf_generated",
    }
    EXTERNAL_ROLES = {
        "retrieved",
        "ret",
        "context",
        "document",
        "rag",
        "search_result",
        "knowledge",
        "tool_output",
        "function_output",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.upstream_base_url.rstrip("/"),
                timeout=httpx.Timeout(self.settings.upstream_timeout_seconds),
                headers=(
                    {"Authorization": f"Bearer {self.settings.upstream_api_key}"}
                    if self.settings.upstream_api_key
                    else {}
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            key: value for key, value in message.items() if key not in self.INTERNAL_MESSAGE_KEYS
        }
        role = str(cleaned.get("role", "user")).lower()
        # Custom provenance roles are useful inside the WAF but are not accepted by
        # the OpenAI-compatible chat schema. Keep the spotlight content and use a
        # standard untrusted user role at the provider boundary.
        if role in self.EXTERNAL_ROLES:
            cleaned["role"] = "user"
        elif role == "tool" and not cleaned.get("tool_call_id"):
            # Provider APIs generally require tool_call_id for role=tool.
            cleaned["role"] = "user"
        return cleaned

    def _build_request_body(
        self,
        original: dict[str, Any],
        upstream_messages: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        body = {key: value for key, value in original.items() if key not in self.INTERNAL_BODY_KEYS}
        if upstream_messages is not None:
            body["messages"] = [self._normalize_message(message) for message in upstream_messages]
        elif isinstance(body.get("messages"), list):
            body["messages"] = [self._normalize_message(message) for message in body["messages"]]
        # Ensure model is set
        if "model" not in body or not body["model"]:
            body["model"] = self.settings.upstream_model
        return body

    async def chat_completions(
        self,
        body: dict[str, Any],
        upstream_messages: list[dict[str, Any]] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = self._get_client()
        payload = self._build_request_body(body, upstream_messages)
        headers = {}
        if extra_headers:
            # Forward relevant headers (e.g., OpenAI-Organization)
            for k, v in extra_headers.items():
                lk = k.lower()
                if lk in ("openai-organization", "openai-project", "x-request-id"):
                    headers[k] = v

        logger.info(
            "upstream request", model=payload.get("model"), stream=payload.get("stream", False)
        )
        try:
            resp = await client.post("chat/completions", json=payload, headers=headers)
            return resp
        except httpx.TimeoutException:
            with suppress(Exception):
                metrics.upstream_errors.labels(code="timeout").inc()
            raise
        except Exception:
            with suppress(Exception):
                metrics.upstream_errors.labels(code="error").inc()
            raise

    async def chat_completions_stream(
        self,
        body: dict[str, Any],
        upstream_messages: list[dict[str, Any]] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[bytes, None]:
        client = self._get_client()
        payload = self._build_request_body(body, upstream_messages)
        payload["stream"] = True
        headers = {}
        if extra_headers:
            for k, v in extra_headers.items():
                lk = k.lower()
                if lk in ("openai-organization", "openai-project", "x-request-id"):
                    headers[k] = v

        async with client.stream("POST", "chat/completions", json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body_bytes = await resp.aread()
                # Yield error as SSE-like? Instead raise to let caller handle
                raise httpx.HTTPStatusError(
                    f"upstream error {resp.status_code}: "
                    f"{body_bytes.decode(errors='ignore')[:2000]}",
                    request=resp.request,
                    response=resp,
                )
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def completions(
        self,
        body: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = self._get_client()
        headers = self._forward_headers(extra_headers)
        return await client.post("completions", json=body, headers=headers)

    async def completions_stream(
        self,
        body: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[bytes, None]:
        client = self._get_client()
        headers = self._forward_headers(extra_headers)
        async with client.stream("POST", "completions", json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                body_bytes = await resp.aread()
                raise httpx.HTTPStatusError(
                    f"upstream error {resp.status_code}: "
                    f"{body_bytes.decode(errors='ignore')[:2000]}",
                    request=resp.request,
                    response=resp,
                )
            async for chunk in resp.aiter_bytes():
                yield chunk

    @staticmethod
    def _forward_headers(extra_headers: dict[str, str] | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if extra_headers:
            for key, value in extra_headers.items():
                if key.lower() in ("openai-organization", "openai-project", "x-request-id"):
                    headers[key] = value
        return headers


# Singleton for app lifespan
_upstream: UpstreamClient | None = None


def get_upstream(settings: Settings | None = None) -> UpstreamClient:
    global _upstream
    if _upstream is None:
        from src.config.settings import get_settings

        _upstream = UpstreamClient(settings or get_settings())
    return _upstream


def reset_upstream() -> None:
    global _upstream
    _upstream = None
