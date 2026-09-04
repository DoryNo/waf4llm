from __future__ import annotations

import httpx
import pytest

from src.config.settings import reset_settings_cache
from src.observability.alerts import (
    AlertManager,
    get_alert_manager,
    reset_alert_manager,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_should_alert_respects_min_level():
    manager = AlertManager("http://hook.test/x", min_level="block")
    assert manager.should_alert("block") is True
    assert manager.should_alert("exclude") is False
    assert manager.should_alert("log") is False


async def test_should_alert_false_without_url():
    manager = AlertManager("", min_level="block")
    assert manager.should_alert("block") is False


async def test_send_alert_posts_slack_payload():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read()
        return httpx.Response(200)

    manager = AlertManager("http://hook.test/x", min_level="block", client=_client(handler))
    ok = await manager.send_alert(
        "block", {"request_id": "req-1", "reason": "heuristic", "confidence": 0.9}
    )
    await manager.close()

    assert ok is True
    assert seen["url"] == "http://hook.test/x"
    body = httpx.Response(200, content=seen["json"]).json()
    assert "WAF BLOCK" in body["text"]
    assert body["waf_alert"]["request_id"] == "req-1"


async def test_send_alert_skips_below_min_level():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200)

    manager = AlertManager("http://hook.test/x", min_level="block", client=_client(handler))
    ok = await manager.send_alert("sanitize", {"request_id": "r"})
    await manager.close()

    assert ok is False
    assert calls == []


async def test_send_alert_tolerates_webhook_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    manager = AlertManager("http://hook.test/x", min_level="block", client=_client(handler))
    ok = await manager.send_alert("block", {"request_id": "r"})
    await manager.close()
    assert ok is False


async def test_alert_block_cooldown_suppresses_burst(monkeypatch):
    from src.observability import alerts as alerts_mod

    monkeypatch.setattr(alerts_mod, "_ALERT_COOLDOWN_SECONDS", 60.0)
    fired: list[tuple[str, dict]] = []

    class FakeManager:
        def should_alert(self, level: str) -> bool:
            return True

        async def send_alert(self, level: str, payload: dict) -> bool:
            fired.append((level, payload))
            return True

    monkeypatch.setattr(alerts_mod, "get_alert_manager", lambda: FakeManager())
    alerts_mod._alert_state.clear()

    alerts_mod.alert_block("req-a", {"reason": "x"})
    alerts_mod.alert_block("req-b", {"reason": "y"})  # inside cooldown -> skipped
    await asyncio_wait_for_tasks()
    assert len(fired) == 1
    assert fired[0][1]["request_id"] == "req-a"


async def asyncio_wait_for_tasks() -> None:
    import asyncio

    await asyncio.sleep(0)


async def test_get_alert_manager_disabled_has_no_url(monkeypatch):
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.setenv("ALERTS_WEBHOOK_URL", "http://hook.test/x")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_settings_cache()
    reset_alert_manager()
    try:
        manager = get_alert_manager()
        assert manager.should_alert("block") is False
    finally:
        reset_settings_cache()
        reset_alert_manager()


@pytest.mark.parametrize("enabled", ["true"])
async def test_get_alert_manager_enabled(monkeypatch, enabled):
    monkeypatch.setenv("ALERTS_ENABLED", enabled)
    monkeypatch.setenv("ALERTS_WEBHOOK_URL", "http://hook.test/x")
    monkeypatch.setenv("ALERTS_MIN_LEVEL", "exclude")
    reset_settings_cache()
    reset_alert_manager()
    try:
        manager = get_alert_manager()
        assert manager.webhook_url == "http://hook.test/x"
        assert manager.min_level == "exclude"
        assert manager.should_alert("block") is True
        assert manager.should_alert("log") is False
    finally:
        reset_settings_cache()
        reset_alert_manager()


async def test_alert_block_no_running_loop_does_not_raise():
    # fire_and_forget must tolerate being called outside an event loop
    from src.observability.alerts import fire_and_forget

    manager = AlertManager("", min_level="block")
    fire_and_forget(manager, "block", {"request_id": "r"})  # no url -> no-op anyway
