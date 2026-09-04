from __future__ import annotations

import dataclasses
import enum
import time
import uuid
from typing import Any, Protocol


class DecisionLevel(enum.StrEnum):
    allow = "allow"
    log = "log"
    sanitize = "sanitize"
    exclude = "exclude"
    block = "block"


class ProvenanceTag(enum.StrEnum):
    SYS = "SYS"
    USR = "USR"
    RET = "RET"
    TOOL = "TOOL"


def decision_level_for_confidence(
    confidence: float,
    *,
    threshold_log: float = 0.3,
    threshold_sanitize: float = 0.55,
    threshold_exclude: float = 0.8,
    threshold_block: float = 0.92,
) -> DecisionLevel:
    """Map a normalized confidence score to the graduated response level."""
    confidence = max(0.0, min(1.0, confidence))
    if confidence >= threshold_block:
        return DecisionLevel.block
    if confidence >= threshold_exclude:
        return DecisionLevel.exclude
    if confidence >= threshold_sanitize:
        return DecisionLevel.sanitize
    if confidence >= threshold_log:
        return DecisionLevel.log
    return DecisionLevel.allow


@dataclasses.dataclass
class TaggedSegment:
    tag: ProvenanceTag
    content: str
    index: int
    role: str
    normalized_content: str | None = None

    def effective_content(self) -> str:
        return self.normalized_content if self.normalized_content is not None else self.content


@dataclasses.dataclass
class LayerResult:
    layer: str
    passed: bool
    confidence: float = 0.0
    level: DecisionLevel = DecisionLevel.allow
    reason: str | None = None
    trigger_tokens: list[str] | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineContext:
    request_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    raw_body: dict[str, Any] = dataclasses.field(default_factory=dict)
    headers: dict[str, str] = dataclasses.field(default_factory=dict)

    # Populated by provenance
    segments: list[TaggedSegment] = dataclasses.field(default_factory=list)

    # Populated by normalizer (mutates segments.normalized_content)
    normalized: bool = False

    # Accumulated results
    layer_results: list[LayerResult] = dataclasses.field(default_factory=list)

    # Decision (updated incrementally, finalized by DecisionEngine)
    decision: DecisionLevel = DecisionLevel.allow
    confidence: float = 0.0
    block_reason: str | None = None

    # Output guard
    canary_token: str | None = None
    canary_hit: bool = False

    # Sanitized messages to send upstream (if None, use original)
    upstream_messages: list[dict[str, Any]] | None = None

    # Timing
    started_at: float = dataclasses.field(default_factory=time.monotonic)

    # Misc (e.g., nonce for spotlighting)
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Route that created this context (used for request logging)
    route: str = "/v1/chat/completions"

    def add_result(self, result: LayerResult) -> None:
        self.layer_results.append(result)
        # Track max confidence
        if result.confidence > self.confidence:
            self.confidence = result.confidence

    @property
    def messages(self) -> list[dict[str, Any]]:
        messages = self.raw_body.get("messages", [])
        return messages if isinstance(messages, list) else []

    def latency_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)


class PipelineLayer(Protocol):
    name: str

    async def process(self, ctx: PipelineContext) -> LayerResult: ...
