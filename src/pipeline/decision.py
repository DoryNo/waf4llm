from __future__ import annotations

from src.config.settings import Settings
from src.pipeline.base import (
    DecisionLevel,
    LayerResult,
    PipelineContext,
    decision_level_for_confidence,
)
from src.pipeline.provenance import spotlight_wrap_messages


class DecisionEngine:
    """Graduated response: aggregates layer results into final decision."""

    name = "decision"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _level_for_confidence(self, conf: float) -> DecisionLevel:
        return decision_level_for_confidence(
            conf,
            threshold_log=self.settings.threshold_log,
            threshold_sanitize=self.settings.threshold_sanitize,
            threshold_exclude=self.settings.threshold_exclude,
            threshold_block=self.settings.threshold_block,
        )

    def decide(self, ctx: PipelineContext) -> LayerResult:
        # Find highest-severity result
        # Severity order: block > exclude > sanitize > log > allow
        severity_order = {
            DecisionLevel.block: 4,
            DecisionLevel.exclude: 3,
            DecisionLevel.sanitize: 2,
            DecisionLevel.log: 1,
            DecisionLevel.allow: 0,
        }

        best: LayerResult | None = None
        best_level = DecisionLevel.allow
        best_sev = -1
        best_conf = 0.0

        for r in ctx.layer_results:
            # Detector implementations may use their own coarse labels. Re-map
            # confidence at the policy boundary so configurable thresholds are
            # applied consistently across heuristic and classifier layers.
            level = (
                DecisionLevel.block
                if r.level == DecisionLevel.block
                else self._level_for_confidence(r.confidence)
            )
            sev = severity_order.get(level, 0)
            # Prefer higher severity, break ties by confidence
            if sev > best_sev or (sev == best_sev and r.confidence > best_conf):
                best = r
                best_level = level
                best_sev = sev
                best_conf = r.confidence

        if best is None or best_sev == 0:
            ctx.decision = DecisionLevel.allow
            ctx.confidence = 0.0
            return LayerResult(
                layer=self.name,
                passed=True,
                confidence=0.0,
                level=DecisionLevel.allow,
                reason="no detections",
            )

        ctx.decision = best_level
        ctx.confidence = best.confidence
        if best_level == DecisionLevel.block:
            ctx.block_reason = best.reason

        return LayerResult(
            layer=self.name,
            passed=best_level != DecisionLevel.block,
            confidence=best.confidence,
            level=best_level,
            reason=f"decision={best_level.value} from {best.layer}: {best.reason}",
            trigger_tokens=best.trigger_tokens,
            extra={"source_layer": best.layer},
        )

    def apply_sanitization(self, ctx: PipelineContext) -> None:
        """Mutate ctx.upstream_messages based on decision.

        - sanitize: wrap flagged RET segments with spotlight delimiters
        - exclude: remove flagged segment entirely
        - block: handled upstream (raise), no mutation needed
        - allow/log: no mutation
        """
        if ctx.decision not in (DecisionLevel.sanitize, DecisionLevel.exclude):
            return

        nonce = ctx.meta.get("spotlight_nonce", "NONCE")
        messages = [dict(m) for m in ctx.messages]

        # Find flagged segment index from heuristic/classifier results
        flagged_idx: int | None = None
        for r in ctx.layer_results:
            if not r.extra:
                continue
            candidate = r.extra.get("flagged_segment", r.extra.get("best_idx"))
            if candidate is not None:
                flagged_idx = int(candidate)
                break

        if ctx.decision == DecisionLevel.exclude:
            if flagged_idx is not None and 0 <= flagged_idx < len(messages):
                # Remove only the flagged message, not the entire context.
                messages.pop(flagged_idx)
            elif flagged_idx is not None:
                # The index can refer to a top-level RAG/tool segment that has not
                # become an upstream message yet.
                excluded = ctx.meta.setdefault("excluded_segment_indices", set())
                if isinstance(excluded, set):
                    excluded.add(flagged_idx)
            ctx.upstream_messages = messages
            return

        if ctx.decision == DecisionLevel.sanitize:
            # Use the same idempotent wrapper as the always-on snapshot defense.
            # For a message-level finding, wrap that message; remaining external
            # messages are wrapped later by always_spotlight_ret_tool().
            segment = next((s for s in ctx.segments if s.index == flagged_idx), None)
            should_target_one = segment is not None and segment.tag.value in ("RET", "TOOL")
            ctx.upstream_messages = spotlight_wrap_messages(
                messages,
                nonce,
                flagged_idx=flagged_idx if should_target_one else None,
            )

    async def process(self, ctx: PipelineContext) -> LayerResult:
        result = self.decide(ctx)
        # Apply sanitization side-effect
        if result.level in (DecisionLevel.sanitize, DecisionLevel.exclude):
            self.apply_sanitization(ctx)
        return result
