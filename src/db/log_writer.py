from __future__ import annotations

import json
from typing import Any

from src.db.models import Detection, RequestLog
from src.observability.logging import get_logger

logger = get_logger("db.log")

_MAX_BODY_CHARS = 8192


def _dump(value: Any, limit: int = _MAX_BODY_CHARS) -> str | None:
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


async def persist_context(ctx: Any) -> None:
    """Store a RequestLog row + Detection rows for a finished pipeline run.

    Best-effort: telemetry failures must never affect the proxy path.
    """
    from src.db.session import _get_sessionmaker

    try:
        maker = _get_sessionmaker()
        async with maker() as session:
            log = RequestLog(
                id=ctx.request_id,
                path=ctx.route,
                model=str(ctx.raw_body.get("model"))[:128] if ctx.raw_body.get("model") else None,
                decision=ctx.decision.value,
                confidence=ctx.confidence,
                canary_token=ctx.canary_token,
                canary_hit=ctx.canary_hit,
                latency_ms=ctx.latency_ms(),
                request_body=_dump(ctx.raw_body),
            )
            session.add(log)
            for result in ctx.layer_results:
                if result.level.value == "allow":
                    continue
                session.add(
                    Detection(
                        request_id=ctx.request_id,
                        layer=result.layer,
                        level=result.level.value,
                        confidence=result.confidence,
                        trigger_tokens=_dump(result.trigger_tokens),
                        detail=result.extra or None,
                    )
                )
            await session.commit()
    except Exception as e:
        logger.warning("request log write failed", error=str(e), request_id=ctx.request_id)


async def persist_attribution(
    request_id: str, entry: dict[str, Any], score: float | None = None
) -> None:
    """Store one token-level attribution row (Phase 8.3). Best-effort."""
    from src.db.models import Attribution
    from src.db.session import _get_sessionmaker

    try:
        maker = _get_sessionmaker()
        async with maker() as session:
            session.add(
                Attribution(
                    request_id=request_id,
                    layer=entry.get("layer", "classifier"),
                    method=entry.get("method", "attention"),
                    model=entry.get("model"),
                    score=score,
                    text=entry.get("text"),
                    tokens=entry.get("tokens") or None,
                )
            )
            await session.commit()
    except Exception as e:
        logger.warning("attribution write failed", error=str(e), request_id=request_id)


async def mark_canary_hit(ctx: Any) -> None:
    """Update the stored request after a post-inference canary block."""
    from sqlalchemy import update

    from src.db.session import _get_sessionmaker

    try:
        maker = _get_sessionmaker()
        async with maker() as session:
            await session.execute(
                update(RequestLog)
                .where(RequestLog.id == ctx.request_id)
                .values(canary_hit=True, decision="block", confidence=1.0)
            )
            await session.commit()
    except Exception as e:
        logger.warning("canary hit update failed", error=str(e), request_id=ctx.request_id)
