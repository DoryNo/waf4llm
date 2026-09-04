from __future__ import annotations

from src.config.settings import Settings
from src.proxy.upstream import UpstreamClient


def make_client() -> UpstreamClient:
    settings = Settings(
        _env_file=None,
        DATABASE_URL="sqlite+aiosqlite:///./test.db",
        REDIS_URL="redis://localhost:6379/0",
        UPSTREAM_BASE_URL="https://example.test/v1",
    )
    return UpstreamClient(settings)


def test_build_request_body_strips_waf_metadata_and_custom_roles():
    client = make_client()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "raw"}],
        "rag_chunks": [{"content": "must not be sent as a raw body field"}],
        "context": "also internal",
    }
    messages = [
        {
            "role": "system",
            "content": "security",
            "metadata": {"tenant": "internal"},
        },
        {
            "role": "retrieved",
            "content": "<<<DATA-N>>>doc<<<END-DATA-N>>>",
            "provenance": "RET",
            "source": "retrieval",
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "valid tool output",
            "_waf_generated": False,
        },
        {"role": "tool", "content": "tool without call id"},
    ]

    payload = client._build_request_body(body, messages)

    assert "rag_chunks" not in payload
    assert "context" not in payload
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "tool",
        "user",
    ]
    assert "provenance" not in payload["messages"][1]
    assert "source" not in payload["messages"][1]
    assert "metadata" not in payload["messages"][0]
    assert payload["messages"][2]["tool_call_id"] == "call_1"
    assert "_waf_generated" not in payload["messages"][2]


def test_build_request_body_preserves_standard_provider_fields():
    client = make_client()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello", "name": "caller"}],
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    payload = client._build_request_body(body, None)

    assert payload["temperature"] == 0.2
    assert payload["stream"] is False
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"] == [{"role": "user", "content": "hello", "name": "caller"}]


def test_forward_headers_only_allows_provider_metadata():
    headers = UpstreamClient._forward_headers(
        {
            "Authorization": "attacker-token",
            "OpenAI-Organization": "org-test",
            "X-Request-Id": "request-test",
            "Cookie": "secret=1",
        }
    )

    assert headers == {
        "OpenAI-Organization": "org-test",
        "X-Request-Id": "request-test",
    }
