from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from src.observability.logging import get_logger

logger = get_logger("alerts")

# Severity used to compare against alerts_min_level.
_LEVEL_ORDER = {"log": 0, "sanitize": 1, "exclude": 2, "block": 3}


class AlertManager:
    """Phase 9.3 — fire webhook/Slack alerts on critical blocks.

    The payload is Slack-compatible (`text` renders in Slack) and carries
    structured fields for generic webhook consumers.
    """

    def __init__(
        self,
        webhook_url: str,
        min_level: str = "block",
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.webhook_url = webhook_url.strip()
        self.min_level = min_level.strip().lower()
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._own_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    def should_alert(self, level: str) -> bool:
        if not self.webhook_url:
            return False
        return _LEVEL_ORDER.get(level.lower(), 0) >= _LEVEL_ORDER.get(self.min_level, 3)

    async def send_alert(self, level: str, payload: dict[str, Any]) -> bool:
        """POST an alert to the webhook. Returns True when delivery succeeded."""
        if not self.should_alert(level):
            return False
        client = self._get_client()
        try:
            resp = await client.post(self.webhook_url, json=self.format_message(level, payload))
            if resp.status_code >= 400:
                logger.warning("alert webhook rejected", status=resp.status_code, level=level)
                return False
            return True
        except Exception as e:
            logger.warning("alert webhook failed", error=str(e), level=level)
            return False

    @staticmethod
    def format_message(level: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("request_id", "unknown")
        reason = payload.get("reason") or "n/a"
        confidence = payload.get("confidence")
        text = (
            f":rotating_light: WAF {level.upper()} — request_id={request_id} "
            f"confidence={confidence} reason={reason}"
        )
        return {"text": text, "waf_alert": {"level": level, **payload}}

    async def close(self) -> None:
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def fire_and_forget(manager: AlertManager, level: str, payload: dict[str, Any]) -> None:
    """Schedule an alert without blocking the request path; errors are logged."""

    async def _run() -> None:
        try:
            await manager.send_alert(level, payload)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("alert task failed", error=str(e), level=level)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("no running loop; alert skipped", level=level)
        return
    loop.create_task(_run())


_manager: AlertManager | None = None
_manager_config: tuple[str, str, float] | None = None


def get_alert_manager() -> AlertManager:
    """Process-wide manager built from settings; rebuilt if settings change."""
    global _manager, _manager_config
    from src.config.settings import get_settings

    settings = get_settings()
    config = (
        settings.alerts_webhook_url,
        settings.alerts_min_level,
        settings.alerts_timeout_seconds,
    )
    if _manager is None or _manager_config != config:
        _manager = AlertManager(
            webhook_url=settings.alerts_webhook_url if settings.alerts_enabled else "",
            min_level=settings.alerts_min_level,
            timeout_seconds=settings.alerts_timeout_seconds,
        )
        _manager_config = config
    return _manager


def reset_alert_manager() -> None:
    global _manager, _manager_config
    _manager = None
    _manager_config = None


_alert_state: dict[str, float] = {}
_ALERT_COOLDOWN_SECONDS = 30.0


def alert_block(request_id: str, payload: dict[str, Any]) -> None:
    """Public hook: fire a 'block' alert (rate-limited per request_id prefix)."""
    manager = get_alert_manager()
    if not manager.should_alert("block"):
        return
    now = time.monotonic()
    last = _alert_state.get("last")
    # Cooldown protects the webhook from bursts (e.g. scripted attacks).
    if last is not None and now - last < _ALERT_COOLDOWN_SECONDS:
        return
    _alert_state["last"] = now
    fire_and_forget(manager, "block", {"request_id": request_id, **payload})
