from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any

from src.config.settings import Settings, get_settings
from src.db.log_writer import mark_canary_hit, persist_attribution, persist_context
from src.observability.alerts import alert_block
from src.observability.logging import get_logger
from src.observability.metrics import metrics
from src.pipeline.base import DecisionLevel, LayerResult, PipelineContext
from src.pipeline.classifier import ClassifierLayer
from src.pipeline.decision import DecisionEngine
from src.pipeline.heuristic import HeuristicLayer
from src.pipeline.normalizer import NormalizerLayer
from src.pipeline.output_guard import OutputGuardLayer
from src.pipeline.provenance import ProvenanceLayer, always_spotlight_ret_tool
from src.retrain.collector import collect_from_context

logger = get_logger("pipeline")


def _schedule(coro: Any) -> None:
    """Schedule a coroutine on the running loop if there is one (best-effort)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        return
    loop.create_task(coro)


class PipelineOrchestrator:
    """Runs layers sequentially: provenance -> normalizer -> heuristic -> classifier -> decision -> output_guard (pre)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provenance = ProvenanceLayer()
        self.normalizer = NormalizerLayer()
        self.heuristic = HeuristicLayer(settings=self.settings)
        self.classifier = ClassifierLayer(
            enabled=self.settings.enable_classifier,
            settings=self.settings,
        )
        self.decision = DecisionEngine(self.settings)
        self.output_guard = OutputGuardLayer(
            enabled=self.settings.enable_output_guard and self.settings.canary_enabled,
            canary_length=self.settings.canary_length,
        )

    def _should_run(self, layer_name: str) -> bool:
        mapping = {
            "provenance": self.settings.enable_provenance,
            "normalizer": self.settings.enable_normalizer,
            "heuristic": self.settings.enable_heuristic,
            "classifier": self.settings.enable_classifier,
            "output_guard": self.settings.enable_output_guard,
        }
        return mapping.get(layer_name, True)

    async def _run_layer(self, layer: Any, ctx: PipelineContext) -> LayerResult:
        name = getattr(layer, "name", layer.__class__.__name__)
        if not self._should_run(name):
            return LayerResult(
                layer=name, passed=True, confidence=0.0, reason="disabled via config"
            )

        start = time.monotonic()
        try:
            result: LayerResult = await layer.process(ctx)
            ctx.add_result(result)

            # Update metrics
            with suppress(Exception):
                metrics.pipeline_layer_duration.labels(layer=name).observe(time.monotonic() - start)
                if result.level != DecisionLevel.allow:
                    metrics.detections_total.labels(level=result.level.value, layer=name).inc()

            # Graduated response short-circuit on block (except decision/output_guard handles block)
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            with suppress(Exception):
                metrics.pipeline_layer_duration.labels(layer=name).observe(elapsed)

            logger.error(
                "pipeline layer failed", layer=name, error=str(e), request_id=ctx.request_id
            )

            # Fail-open vs fail-closed
            if self.settings.fail_mode.value == "closed":
                # Treat as block
                fail_result = LayerResult(
                    layer=name,
                    passed=False,
                    confidence=1.0,
                    level=DecisionLevel.block,
                    reason=f"layer error (fail-closed): {e}",
                )
                ctx.add_result(fail_result)
                return fail_result
            else:
                # Fail-open: log and continue
                fail_result = LayerResult(
                    layer=name,
                    passed=True,
                    confidence=0.0,
                    level=DecisionLevel.allow,
                    reason=f"layer error (fail-open): {e}",
                    extra={"error": str(e)},
                )
                ctx.add_result(fail_result)
                return fail_result

    async def run_pre_inference(self, ctx: PipelineContext) -> PipelineContext:
        """Run all pre-inference layers. Returns ctx with decision and possibly mutated upstream_messages."""
        # Order matters — provenance must be first (sets nonce), normalizer second, decision before output_guard
        layers: list[Any] = [
            self.provenance,
            self.normalizer,
            self.heuristic,
            self.classifier,
            self.decision,
            self.output_guard,
        ]

        for layer in layers:
            result = await self._run_layer(layer, ctx)
            if result.level == DecisionLevel.block and layer is not self.output_guard:
                # Ensure decision is finalized
                if layer.name != "decision" and not any(
                    r.layer == "decision" for r in ctx.layer_results
                ):
                    await self._run_layer(self.decision, ctx)
                # On block we don't need output_guard/canary — request won't reach LLM
                break

        # Defensive: ensure decision ran
        if not any(r.layer == "decision" for r in ctx.layer_results):
            await self._run_layer(self.decision, ctx)

        # Phase 2.2 — Always-on spotlighting (snapshot defense) for RET/TOOL even on allow/log
        # Must run after decision (so decision's sanitize/exclude logic already applied) but before upstream
        if ctx.decision != DecisionLevel.block:
            try:
                always_spotlight_ret_tool(ctx)
            except Exception as e:
                logger.warning("always-spotlight failed", error=str(e), request_id=ctx.request_id)
                if self.settings.fail_mode.value == "closed":
                    ctx.decision = DecisionLevel.block
                    ctx.confidence = 1.0
                    ctx.block_reason = f"spotlight error (fail-closed): {e}"

        # Output guard canary injection is already done inside output_guard layer when it ran.
        # If output_guard was skipped due to block, no canary needed.

        # Phase 9 — best-effort persistence of the pipeline outcome (dashboard data).
        if self.settings.log_requests_enabled:
            await persist_context(ctx)

        # Phase 10.1 — best-effort collection of borderline/blocked cases.
        if self.settings.retrain_enabled:
            _schedule(collect_from_context(ctx, self.settings))

        # Phase 8.3 — best-effort persistence of token-level attributions.
        if self.settings.xai_store_enabled and ctx.meta.get("xai"):
            for entry in ctx.meta["xai"]:
                await persist_attribution(ctx.request_id, entry, score=ctx.confidence)

        # Phase 9.3 — webhook/Slack alert on blocks.
        if ctx.decision == DecisionLevel.block:
            alert_block(
                ctx.request_id,
                {
                    "reason": ctx.block_reason,
                    "confidence": round(ctx.confidence, 4),
                    "layer": next(
                        (r.layer for r in ctx.layer_results if r.level == DecisionLevel.block),
                        None,
                    ),
                },
            )

        return ctx

    def check_output(self, ctx: PipelineContext, output_text: str) -> LayerResult:
        """Post-inference canary check."""
        result = self.output_guard.check_output(ctx, output_text)
        ctx.add_result(result)
        if not result.passed:
            with suppress(Exception):
                metrics.canary_hits.inc()
            # Elevate decision to block if canary hit
            ctx.decision = DecisionLevel.block
            ctx.confidence = max(ctx.confidence, 1.0)
            ctx.block_reason = result.reason

            # Phase 9 — persist canary outcome + alert (fire-and-forget).
            if self.settings.log_requests_enabled:
                _schedule(mark_canary_hit(ctx))
            alert_block(
                ctx.request_id,
                {
                    "reason": result.reason,
                    "confidence": 1.0,
                    "layer": "output_guard",
                    "canary_hit": True,
                },
            )
        return result


_orchestrator: PipelineOrchestrator | None = None


def get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    global _orchestrator
    _orchestrator = None
