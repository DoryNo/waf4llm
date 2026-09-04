from __future__ import annotations

import asyncio
import importlib
import math
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from src.config.settings import FailMode, Settings, get_settings
from src.observability.logging import get_logger
from src.pipeline.base import (
    DecisionLevel,
    LayerResult,
    PipelineContext,
    ProvenanceTag,
    decision_level_for_confidence,
)
from src.pipeline.multiturn import build_multiturn_state
from src.pipeline.xai import AttributionResult, build_token_attribution, cls_attention_scores

logger = get_logger("classifier")


class ClassifierLayer:
    """BERT classifier wrapper — Phase 5.

    Features:
    - Lazy load from HuggingFace (Prompt Guard 2 / DeBERTa) or ONNX
    - Per-tag scoring (USR/RET separately) with tag multipliers
    - Per-turn aggregation + multiturn sliding window (crescendo)
    - Batching via tokenizer batch_encode
    - Graceful fallback when torch/transformers unavailable
    """

    name = "classifier"

    def __init__(
        self,
        enabled: bool | None = None,
        model_path: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        device: str | None = None,
        onnx_path: str | None = None,
        window_size: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.enabled = enabled if enabled is not None else self.settings.enable_classifier
        self.model_path = model_path or self.settings.classifier_model
        self.batch_size = (
            batch_size if batch_size is not None else self.settings.classifier_batch_size
        )
        self.max_length = (
            max_length if max_length is not None else self.settings.classifier_max_length
        )
        self.device = device or self.settings.classifier_device
        self.onnx_path = onnx_path or self.settings.classifier_onnx_path
        self.window_size = (
            window_size if window_size is not None else self.settings.multiturn_window
        )
        self.cumulative_threshold = self.settings.multiturn_cumulative_threshold
        self.fail_mode = self.settings.fail_mode
        # Phase 8.1 — attention-based token importance (hot path, same forward pass)
        self.xai_attention_enabled = self.settings.xai_attention_enabled

        self._tokenizer = None
        self._model = None
        self._onnx_session = None
        self._loaded = False
        self._load_error: str | None = None
        self._is_onnx = False
        self._config = None
        self._load_lock = threading.Lock()

    def _try_load_onnx(self) -> bool:
        if not self.onnx_path:
            return False
        path = Path(self.onnx_path)
        if not path.exists():
            return False
        try:
            import onnxruntime as ort
            from transformers import AutoConfig, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=False
            )
            self._config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=False)
            self._onnx_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._is_onnx = True
            logger.info("classifier ONNX loaded", path=str(path), model=self.model_path)
            return True
        except Exception as e:
            logger.warning("ONNX load failed, falling back to transformers", error=str(e))
            self._onnx_session = None
            return False

    def _try_load_transformers(self) -> bool:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Prefer local cache; don't download in tests unless explicitly enabled
            # Use local_files_only if model not cached? But we want to download if enabled and missing?
            # For MVP we try to load; if fails (no internet, no cache), we fallback gracefully.
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=False, local_files_only=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=False,
                local_files_only=False,
                # SDPA does not return attention weights; XAI needs eager attention
                **({"attn_implementation": "eager"} if self.xai_attention_enabled else {}),
            )
            self._tokenizer = tokenizer
            self._model = model
            self._config = model.config
            self._is_onnx = False
            model.eval()
            if self.device != "cpu":
                with suppress(Exception):
                    model.to(self.device)
            logger.info("classifier transformers loaded", model=self.model_path, device=self.device)
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(
                "classifier transformers load failed", model=self.model_path, error=str(e)
            )
            return False

    def load(self) -> bool:
        if not self.enabled:
            return False
        if self._loaded:
            return self._model is not None or self._onnx_session is not None

        with self._load_lock:
            if self._loaded:
                return self._model is not None or self._onnx_session is not None
            # Mark as attempted only after the load attempt finishes. This keeps state
            # consistent if a loader raises unexpectedly.
            loaded = self._try_load_onnx() or self._try_load_transformers()
            self._loaded = True
            return loaded

    def _failure_result(self, reason: str) -> LayerResult:
        """Return a policy-aware result for model/load/inference failures."""
        closed = self.fail_mode == FailMode.closed
        return LayerResult(
            layer=self.name,
            passed=not closed,
            confidence=1.0 if closed else 0.0,
            level=DecisionLevel.block if closed else DecisionLevel.allow,
            reason=f"{reason} ({'fail-closed' if closed else 'fail-open'})",
            extra={"error": reason, "model": self.model_path},
        )

    def _softmax(self, logits: list[float]) -> list[float]:
        # Numerically stable softmax
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def _injection_probability(self, logits: list[float]) -> float:
        """Convert model logits into probability of malicious/injection content.

        Binary models conventionally use label 1 for injection. For multi-class
        models, sum labels whose names indicate malicious/jailbreak/injection and
        exclude benign labels.
        """
        if not logits:
            return 0.0
        probs = self._softmax(logits)
        if len(probs) == 1:
            return probs[0]

        labels: dict[int, str] = {}
        id2label = getattr(self._config, "id2label", None)
        if isinstance(id2label, dict):
            for key, value in id2label.items():
                try:
                    labels[int(key)] = str(value).lower()
                except (TypeError, ValueError):
                    continue

        malicious_terms = ("inject", "jailbreak", "malicious", "attack", "unsafe", "harmful")
        malicious_indices = {
            index
            for index, label in labels.items()
            if any(term in label for term in malicious_terms)
        }
        if malicious_indices:
            return sum(probs[index] for index in malicious_indices if index < len(probs))

        # Keep the common binary convention as a safe fallback.
        return probs[1]

    def _inference_transformers(
        self, texts: list[str], want_attention: bool = False
    ) -> tuple[list[float], list[dict[str, Any] | None]]:
        # Returns (injection probabilities per text, per-row attention data)
        if self._tokenizer is None or self._model is None:
            return [0.0] * len(texts), [None] * len(texts)
        try:
            import torch

            all_scores: list[float] = []
            all_attention: list[dict[str, Any] | None] = []
            # Batch
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._model(**dict(enc), output_attentions=want_attention)
                    logits = outputs.logits  # shape [batch, num_labels]
                    if want_attention and getattr(outputs, "attentions", None):
                        attentions = list(outputs.attentions)
                        for row in range(len(batch)):
                            # keep the batch dim: cls_attention_scores expects
                            # per-layer tensors shaped [batch, heads, seq, seq]
                            raw = cls_attention_scores(
                                [a[row : row + 1] for a in attentions],
                                enc["attention_mask"][row].tolist(),
                            )
                            token_strings = self._tokenizer.convert_ids_to_tokens(
                                enc["input_ids"][row].tolist()
                            )
                            all_attention.append({"token_strings": token_strings, "raw": raw})
                    else:
                        all_attention.extend([None] * len(batch))
                    for row in logits:
                        all_scores.append(self._injection_probability(row.tolist()))
            return all_scores, all_attention
        except Exception as e:
            logger.error("transformers inference failed", error=str(e))
            raise RuntimeError(f"transformers inference failed: {e}") from e

    def _inference_onnx(self, texts: list[str]) -> tuple[list[float], list[dict[str, Any] | None]]:
        if self._tokenizer is None or self._onnx_session is None:
            return [0.0] * len(texts), [None] * len(texts)
        try:
            np = importlib.import_module("numpy")

            all_scores: list[float] = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np",
                )
                # ONNX expects input_ids, attention_mask
                ort_inputs = {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                }
                # Some models also need token_type_ids
                if "token_type_ids" in enc:
                    ort_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
                logits = self._onnx_session.run(None, ort_inputs)[0]  # [batch, num_labels]
                for row in logits:
                    all_scores.append(self._injection_probability(row.tolist()))
            return all_scores, [None] * len(texts)
        except Exception as e:
            logger.error("ONNX inference failed", error=str(e))
            raise RuntimeError(f"ONNX inference failed: {e}") from e

    def _run_batch(
        self, texts: list[str], want_attention: bool = False
    ) -> tuple[list[float], list[dict[str, Any] | None]]:
        if self._is_onnx and self._onnx_session is not None:
            return self._inference_onnx(texts)
        return self._inference_transformers(texts, want_attention=want_attention)

    async def _score_segments(
        self, segments, want_attention: bool = False
    ) -> tuple[list[float], list[dict[str, Any] | None]]:
        # Run in thread pool to avoid blocking event loop
        texts = [s.effective_content() for s in segments]
        # Filter empty
        # Replace empty with placeholder to keep alignment
        texts = [t if t.strip() else "[EMPTY]" for t in texts]
        loop = asyncio.get_running_loop()
        scores, attention_rows = await loop.run_in_executor(
            None, lambda: self._run_batch(texts, want_attention=want_attention)
        )
        if len(scores) != len(segments):
            raise RuntimeError(
                f"classifier returned {len(scores)} scores for {len(segments)} segments"
            )
        return scores, attention_rows

    def _aggregate_per_tag(
        self, segments, scores: list[float]
    ) -> tuple[float, dict, int | None, int | None]:
        # Apply tag multipliers and find best segment
        tag_multipliers = {
            ProvenanceTag.USR.value: 1.0,
            ProvenanceTag.RET.value: 0.85,  # RET is still high risk but slightly less than direct user injection
            ProvenanceTag.TOOL.value: 0.7,
            ProvenanceTag.SYS.value: 0.1,  # SYS should rarely be injection
        }
        best = 0.0
        best_idx: int | None = None
        best_pos: int | None = None
        per_tag_max: dict[str, float] = {}
        for pos, (seg, raw_score) in enumerate(zip(segments, scores, strict=True)):
            mult = tag_multipliers.get(seg.tag.value, 1.0)
            adj = max(0.0, min(1.0, float(raw_score))) * mult
            per_tag_max[seg.tag.value] = max(per_tag_max.get(seg.tag.value, 0.0), adj)
            if adj > best:
                best = adj
                best_idx = seg.index
                best_pos = pos
        return best, per_tag_max, best_idx, best_pos

    def export_onnx(self, output_path: str, opset_version: int = 14) -> str:
        """Export current HF model to ONNX (requires torch+transformers+onnx)."""
        if self._model is None and not self._try_load_transformers():
            raise RuntimeError(f"cannot export, model not loaded: {self._load_error}")
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("cannot export, model or tokenizer is not loaded")
        try:
            import torch

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Use optimum export if available, else fallback to torch.onnx
            # Simple torch.onnx export
            dummy = tokenizer("hello world", return_tensors="pt")
            torch.onnx.export(
                model,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(output),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch"},
                },
                opset_version=opset_version,
            )
            logger.info("ONNX exported", path=str(output))
            return str(output)
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}") from e

    async def process(self, ctx: PipelineContext) -> LayerResult:
        if not self.enabled:
            return LayerResult(layer=self.name, passed=True, confidence=0.0, reason="disabled")

        if not self._loaded:
            # Model loading is synchronous and can involve disk/network I/O; keep it
            # off the event loop on the first request.
            loaded = await asyncio.to_thread(self.load)
            if not loaded:
                return self._failure_result(f"model unavailable: {self._load_error or 'unknown'}")

        if not ctx.segments:
            return LayerResult(layer=self.name, passed=True, confidence=0.0, reason="no segments")

        # Phase 8.1 — attention importance rides on the same forward pass
        want_attention = (
            self.xai_attention_enabled and not self._is_onnx and self._model is not None
        )

        # Score all segments in batches
        try:
            scores, attention_rows = await self._score_segments(
                ctx.segments, want_attention=want_attention
            )
        except Exception as e:
            logger.error("classifier scoring failed", error=str(e))
            return self._failure_result(f"scoring error: {e}")

        # Per-tag aggregation
        best_score, per_tag_max, best_idx, best_pos = self._aggregate_per_tag(ctx.segments, scores)

        # Multiturn window analysis (crescendo)
        multiturn_state = build_multiturn_state(
            ctx.segments,
            scores,
            window_size=self.window_size,
            cumulative_threshold=self.cumulative_threshold,
        )
        should_escalate = multiturn_state.should_escalate()
        cum_score = multiturn_state.cumulative_score()

        # Final confidence is max of single-best and escalated cumulative
        final_conf = best_score
        escalated = False
        if should_escalate and cum_score > best_score:
            # Escalate to at least 0.75 if multiturn detects crescendo
            final_conf = max(final_conf, min(0.92, cum_score * 0.85))
            escalated = True

        level = decision_level_for_confidence(
            final_conf,
            threshold_log=self.settings.threshold_log,
            threshold_sanitize=self.settings.threshold_sanitize,
            threshold_exclude=self.settings.threshold_exclude,
            threshold_block=self.settings.threshold_block,
        )

        passed = level in (DecisionLevel.allow, DecisionLevel.log)

        # If multiturn escalated, ensure at least sanitize
        if escalated and passed:
            level = DecisionLevel.sanitize
            passed = False

        # Build trigger tokens from top scoring segments (for observability)
        trigger_tokens: list[str] = []
        for seg, s in zip(ctx.segments, scores, strict=True):
            if s >= 0.5:
                # Use first 5 words as trigger preview
                preview = " ".join(seg.effective_content().split()[:5])
                trigger_tokens.append(f"{seg.tag.value}:{preview}")

        reason = f"classifier max={best_score:.3f} per_tag={per_tag_max} multiturn_cum={cum_score:.3f} escalated={escalated}"

        extra: dict[str, Any] = {
            "per_tag_max": per_tag_max,
            "best_idx": best_idx,
            "raw_scores": [round(s, 4) for s in scores],
            "multiturn": multiturn_state.to_dict(),
            "model": self.model_path,
            "is_onnx": self._is_onnx,
        }

        # Phase 8.1 — attach word-level attention attribution for the best segment
        if want_attention and best_pos is not None and attention_rows[best_pos] is not None:
            try:
                att_row = attention_rows[best_pos]
                seg = ctx.segments[best_pos]
                tokens = build_token_attribution(
                    att_row["token_strings"],  # type: ignore[index]
                    att_row["raw"],  # type: ignore[index]
                    top_k=self.settings.xai_top_k,
                )
                attribution = AttributionResult(
                    method="attention",
                    model=self.model_path,
                    text=seg.effective_content(),
                    tokens=tokens,
                    meta={"segment_tag": seg.tag.value, "segment_index": seg.index},
                )
                extra["xai"] = attribution.to_dict()
                ctx.meta.setdefault("xai", []).append(attribution.to_dict())
            except Exception as e:  # XAI must never break the pipeline
                logger.warning("attention attribution failed", error=str(e))

        return LayerResult(
            layer=self.name,
            passed=passed,
            confidence=float(final_conf),
            level=level,
            reason=reason,
            trigger_tokens=trigger_tokens[:5] if trigger_tokens else None,
            extra=extra,
        )
