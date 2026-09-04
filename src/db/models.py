from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON as GenericJSON
from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    # Request metadata
    method: Mapped[str] = mapped_column(String(16), nullable=False, default="POST")
    path: Mapped[str] = mapped_column(String(512), nullable=False, default="/v1/chat/completions")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    upstream_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Pipeline outcome
    decision: Mapped[str] = mapped_column(
        String(32), nullable=False, default="allow"
    )  # allow|sanitize|exclude|block
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    canary_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    canary_hit: Mapped[bool] = mapped_column(default=False)

    # Latency
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Raw-ish payload (truncated)
    request_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_body: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_request_logs_created_at", "created_at"),)


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    layer: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # provenance|normalizer|heuristic|classifier|output_guard
    level: Mapped[str] = mapped_column(String(32), nullable=False)  # low|medium|high|critical
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    trigger_tokens: Mapped[str | None] = mapped_column(Text, nullable=True)
    detail: Mapped[dict | None] = mapped_column(
        GenericJSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    __table_args__ = (Index("ix_detections_request_id", "request_id"),)


class RetrainItem(Base):
    """One queued example for continuous retraining (Phase 10.1)."""

    __tablename__ = "retrain_queue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)  # injection | benign
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pipeline"
    )  # pipeline | report | corpus
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # pending | consumed | rejected
    meta: Mapped[dict | None] = mapped_column(
        GenericJSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    __table_args__ = (
        Index("ix_retrain_queue_status", "status"),
        Index("ix_retrain_queue_created_at", "created_at"),
    )


class Attribution(Base):
    """Token-level attribution for one request (Phase 8.3)."""

    __tablename__ = "attributions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    layer: Mapped[str] = mapped_column(String(64), nullable=False, default="classifier")
    method: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # attention|integrated_gradients
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tokens: Mapped[dict | None] = mapped_column(
        GenericJSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    __table_args__ = (Index("ix_attributions_request_id", "request_id"),)
