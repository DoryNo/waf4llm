from __future__ import annotations

import uuid
from typing import Any

from src.config.settings import Settings, get_settings
from src.db.models import RetrainItem
from src.observability.logging import get_logger
from src.pipeline.base import DecisionLevel, PipelineContext

logger = get_logger("retrain.collector")

_MAX_TEXT_CHARS = 4096


def _normalize_label(label: str) -> str:
    """Map free-form labels onto the two supported classes."""
    normalized = (label or "").strip().lower()
    if normalized in ("injection", "malicious", "attack", "1", "unsafe"):
        return "injection"
    if normalized in ("benign", "safe", "clean", "0", "ok"):
        return "benign"
    raise ValueError(f"unsupported label: {label!r} (use 'benign' or 'injection')")


def _truncate(text: str) -> str:
    return (text or "").strip()[:_MAX_TEXT_CHARS]


async def enqueue(
    text: str,
    label: str,
    *,
    source: str = "pipeline",
    request_id: str | None = None,
    confidence: float | None = None,
    decision: str | None = None,
    meta: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Add one labeled example to the retrain queue. Best-effort, returns item id.

    Deduplicates against pending items with identical text+label and enforces
    the queue size cap (oldest pending items stay, new ones are dropped).
    """
    cfg = settings or get_settings()
    from src.db.session import _get_sessionmaker

    try:
        clean = _truncate(text)
        if not clean:
            return None
        normalized_label = _normalize_label(label)

        maker = _get_sessionmaker()
        async with maker() as session:
            # dedupe: same pending text+label
            from sqlalchemy import func, select

            dup = await session.execute(
                select(func.count())
                .select_from(RetrainItem)
                .where(
                    RetrainItem.text == clean,
                    RetrainItem.label == normalized_label,
                    RetrainItem.status == "pending",
                )
            )
            if (dup.scalar() or 0) > 0:
                return None

            # queue cap
            total = await session.execute(
                select(func.count()).select_from(RetrainItem).where(RetrainItem.status == "pending")
            )
            if (total.scalar() or 0) >= cfg.retrain_max_queue:
                logger.warning(
                    "retrain queue full, dropping item",
                    max_queue=cfg.retrain_max_queue,
                )
                return None

            item_id = str(uuid.uuid4())
            session.add(
                RetrainItem(
                    id=item_id,
                    text=clean,
                    label=normalized_label,
                    source=source,
                    request_id=request_id,
                    confidence=confidence,
                    decision=decision,
                    status="pending",
                    meta=meta,
                )
            )
            await session.commit()
            return item_id
    except Exception as e:
        logger.warning("retrain enqueue failed", error=str(e))
        return None


def _extract_texts(ctx: PipelineContext) -> list[str]:
    """User-provided segment texts (normalized content when available).

    SYS segments are skipped: the system prompt is trusted content and must
    not end up in the injection corpus.
    """
    from src.pipeline.base import ProvenanceTag

    return [
        seg.effective_content()
        for seg in ctx.segments
        if seg.tag != ProvenanceTag.SYS and seg.effective_content().strip()
    ]


def wants_borderline(ctx: PipelineContext, settings: Settings) -> bool:
    """True when the pipeline outcome is worth collecting for retraining."""
    if not settings.retrain_enabled:
        return False
    if ctx.decision == DecisionLevel.allow:
        return False
    if ctx.decision == DecisionLevel.log:
        return ctx.confidence >= settings.retrain_borderline_min
    # sanitize/exclude/block are always candidates
    return True


async def collect_from_context(ctx: PipelineContext, settings: Settings | None = None) -> int:
    """Queue flagged request segments as injection examples. Best-effort."""
    cfg = settings or get_settings()
    if not wants_borderline(ctx, cfg):
        return 0
    texts = _extract_texts(ctx)
    if not texts:
        return 0
    count = 0
    for text in texts[:8]:  # cap per-request fanout
        result = await enqueue(
            text,
            "injection",
            source="pipeline",
            request_id=ctx.request_id,
            confidence=ctx.confidence,
            decision=ctx.decision.value,
            settings=cfg,
        )
        if result:
            count += 1
    return count


async def report_feedback(
    text: str,
    label: str,
    *,
    request_id: str | None = None,
    confidence: float | None = None,
    settings: Settings | None = None,
) -> str | None:
    """User-reported correction (false positive/negative) — Phase 10.1."""
    return await enqueue(
        text,
        label,
        source="report",
        request_id=request_id,
        confidence=confidence,
        meta={"origin": "user-feedback"},
        settings=settings,
    )
