from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.pipeline.base import ProvenanceTag


@dataclass
class TurnScore:
    index: int
    tag: str
    confidence: float
    text_preview: str  # first 80 chars for debugging


@dataclass
class MultiturnState:
    """Sliding window cumulative risk tracker (crescendo detection).

    Keeps last N turn scores, computes:
    - max single-turn confidence
    - cumulative sum (with decay)
    - count of medium-confidence turns in window
    """

    window_size: int = 6
    decay: float = 0.85  # exponential decay for older turns
    medium_threshold: float = 0.45
    cumulative_threshold: float = 1.2  # sum > threshold => escalate to block

    _window: deque[TurnScore] = field(default_factory=lambda: deque(maxlen=6))

    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in the interval (0, 1]")
        if self.medium_threshold < 0 or self.cumulative_threshold <= 0:
            raise ValueError("risk thresholds must be positive")
        if self._window.maxlen != self.window_size:
            self._window = deque(self._window, maxlen=self.window_size)

    def add(self, score: TurnScore) -> None:
        self._window.append(score)

    def cumulative_score(self) -> float:
        # Weighted sum with decay: most recent has weight 1, older decay
        total = 0.0
        for i, ts in enumerate(reversed(self._window)):
            weight = self.decay**i
            total += ts.confidence * weight
        return total

    def medium_count(self) -> int:
        return sum(1 for ts in self._window if ts.confidence >= self.medium_threshold)

    def should_escalate(self) -> bool:
        # Escalate if cumulative exceeds threshold or 3+ medium turns in window (crescendo)
        if self.cumulative_score() >= self.cumulative_threshold:
            return True
        return self.medium_count() >= 3

    def risk_level(self) -> str:
        cum = self.cumulative_score()
        cnt = self.medium_count()
        critical_threshold = self.cumulative_threshold * 1.25
        medium_threshold = self.cumulative_threshold * 2 / 3
        if cum >= critical_threshold or cnt >= 4:
            return "critical"
        if cum >= self.cumulative_threshold or cnt >= 3:
            return "high"
        if cum >= medium_threshold or cnt >= 2:
            return "medium"
        if cum >= 0.4:
            return "low"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "scores": [
                {"idx": s.index, "tag": s.tag, "conf": round(s.confidence, 3)} for s in self._window
            ],
            "cumulative": round(self.cumulative_score(), 3),
            "medium_count": self.medium_count(),
            "escalate": self.should_escalate(),
            "risk_level": self.risk_level(),
        }


# Global in-memory store per request_id would be needed for cross-request multiturn.
# For MVP we keep per-request window (last N segments within same request).
# For true conversational crescendo, an external store (Redis) would persist across turns.
# Here we implement per-request window; Redis-backed variant can be added later.


def build_multiturn_state(
    segments: Sequence,
    confidences: Sequence[float],
    window_size: int = 6,
    decay: float = 0.85,
    medium_threshold: float = 0.45,
    cumulative_threshold: float = 1.2,
) -> MultiturnState:
    state = MultiturnState(
        window_size=window_size,
        decay=decay,
        medium_threshold=medium_threshold,
        cumulative_threshold=cumulative_threshold,
    )
    for seg, conf in zip(segments, confidences, strict=True):
        # Only consider USR and RET for multiturn risk (SYS is trusted)
        if seg.tag in (ProvenanceTag.USR, ProvenanceTag.RET):
            preview = seg.effective_content()[:80].replace("\n", " ")
            state.add(
                TurnScore(index=seg.index, tag=seg.tag.value, confidence=conf, text_preview=preview)
            )
    return state
