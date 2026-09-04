"""XAI / Explainability — Phase 8.

Two attribution methods:

- Attention-based token importance (8.1): computed from a single classifier
  forward pass with ``output_attentions=True`` — cheap enough for the hot path.
  Importance of a token is the mean attention it receives from the [CLS] token,
  averaged over layers and heads.
- Integrated Gradients via Captum (8.2): detailed attribution, computed on
  demand (async/debug mode) with LayerIntegratedGradients over the embedding
  layer.

Token scores are aggregated from subword pieces to human-readable words and
stored in the ``attributions`` table (8.3) for post-hoc analysis.

Everything degrades gracefully: without torch/transformers/captum the module
raises :class:`XAIUnavailableError` instead of breaking the pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger

logger = get_logger("xai")

_MAX_TEXT_CHARS = 512


class XAIUnavailableError(RuntimeError):
    """Raised when the requested attribution method has no usable backend."""


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise XAIUnavailableError("torch is not installed (pip install torch)") from e
    return torch


# ---------------------------------------------------------------------------
# Pure helpers (no torch required — unit-tested standalone)
# ---------------------------------------------------------------------------


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores into [0, 1]; all-zero input stays all-zero."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


def aggregate_subwords_to_words(
    tokens: list[str], scores: list[float], space_prefixes: tuple[str, ...] = ("▁", "Ġ")
) -> list[tuple[str, float]]:
    """Aggregate subword token scores into word-level scores.

    Handles the two dominant subword conventions:

    - WordPiece (BERT): ``##`` marks a continuation piece; bare pieces start words.
    - SentencePiece: ``▁``/``Ġ`` marks a word start (leading space); other pieces
      are treated as word starts too (CJK/unknown pieces).

    Special tokens (wrapped in ``[...]``) and empty pieces are skipped.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have the same length")

    words: list[str] = []
    word_scores: list[float] = []
    current_word: list[str] = []
    current_score = 0.0

    def flush() -> None:
        nonlocal current_word, current_score
        if current_word:
            words.append("".join(current_word))
            word_scores.append(current_score)
            current_word = []
            current_score = 0.0

    for token, score in zip(tokens, scores, strict=True):
        if not token or (token.startswith("[") and token.endswith("]")):
            continue
        if token.startswith("##"):
            core = token[2:]
            if not core:
                continue
            current_word.append(core)
            current_score += score
        else:
            prefix = next((p for p in space_prefixes if token.startswith(p)), None)
            core = token[len(prefix) :] if prefix else token
            if not core:
                continue
            flush()
            current_word = [core]
            current_score = score
    flush()
    return list(zip(words, word_scores, strict=True))


def top_attributed(words: list[tuple[str, float]], top_k: int = 10) -> list[dict[str, Any]]:
    """Return the ``top_k`` words by score as ``{"token": ..., "score": ...}``."""
    ranked = sorted(words, key=lambda ws: ws[1], reverse=True)[: max(0, top_k)]
    return [{"token": token, "score": round(float(score), 6)} for token, score in ranked]


def build_token_attribution(
    tokens: list[str], scores: list[float], *, top_k: int = 10
) -> list[dict[str, Any]]:
    """Turn raw subword scores into a normalized word-level top-k attribution."""
    normalized = normalize_scores(scores)
    words = aggregate_subwords_to_words(tokens, normalized)
    return top_attributed(words, top_k=top_k)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _TokenizersLike(Protocol):
    """Minimal surface of a HF tokenizer used here."""

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...


def injection_label_index(config: Any) -> int:
    """Pick the class index that represents malicious/injection content."""
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict):
        terms = ("inject", "jailbreak", "malicious", "attack", "unsafe", "harmful")
        for key, label in id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if any(term in str(label).lower() for term in terms):
                return idx
    return 1


def cls_attention_scores(attentions: list[Any], attention_mask: list[int]) -> list[float]:
    """Per-token importance from attention tensors of one encoded row.

    ``attentions`` is the tuple of per-layer tensors shaped
    ``[batch, heads, seq, seq]``. Returns one float per sequence position: the
    attention that position receives from the [CLS] token (index 0), averaged
    over layers and heads. Padding positions get 0.
    """
    torch = _import_torch()
    stacked = torch.stack([a[0] for a in attentions])  # [layers, heads, seq, seq]
    received = stacked.mean(dim=(0, 1))  # [seq, seq]
    cls_row = received[0]  # attention [CLS] -> token
    scores = [float(s) for s in cls_row.tolist()]
    return [
        score if int(mask) == 1 else 0.0 for score, mask in zip(scores, attention_mask, strict=True)
    ]


@dataclass
class AttributionResult:
    """One attribution: model + text + word-level token scores."""

    method: str  # "attention" | "integrated_gradients"
    model: str
    text: str
    tokens: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "text": self.text[:_MAX_TEXT_CHARS],
            "tokens": self.tokens,
            "meta": self.meta,
        }


class XAIManager:
    """Facade over the classifier's model for token attribution.

    - ``explain_attention`` — hot-path method, one forward pass.
    - ``explain_integrated_gradients`` — on-demand, Captum-based.
    - ``explain_text`` — async wrapper dispatching on ``method``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _load_classifier(self) -> tuple[Any, Any, str]:
        """Return (model, tokenizer, model_name) from a fresh ClassifierLayer."""
        from src.pipeline.classifier import ClassifierLayer

        layer = ClassifierLayer(enabled=True, settings=self.settings)
        if not layer.load():
            raise XAIUnavailableError(
                f"classifier model unavailable: {layer._load_error or 'load failed'}"
            )
        if layer._is_onnx or layer._model is None or layer._tokenizer is None:
            raise XAIUnavailableError("XAI requires the transformers backend, not ONNX")
        return layer._model, layer._tokenizer, layer.model_path

    @staticmethod
    def _encode(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
        enc = tokenizer(text[:4096], return_tensors="pt", truncation=True, max_length=max_length)
        enc = {
            k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        return {
            "enc": enc,
            "mask": enc["attention_mask"][0].tolist(),
            "ids": enc["input_ids"][0].tolist(),
        }

    def explain_attention(self, text: str) -> AttributionResult:
        """Attention-based importance in a single forward pass."""
        model, tokenizer, model_name = self._load_classifier()
        torch = _import_torch()
        prepared = self._encode(tokenizer, text, self.settings.classifier_max_length)
        enc, mask = prepared["enc"], prepared["mask"]

        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
        if not getattr(outputs, "attentions", None):
            raise XAIUnavailableError("model does not expose attention weights")

        raw = cls_attention_scores(list(outputs.attentions), mask)
        token_strings: list[str] = tokenizer.convert_ids_to_tokens(prepared["ids"])
        tokens = build_token_attribution(token_strings, raw, top_k=self.settings.xai_top_k)
        return AttributionResult(
            method="attention",
            model=model_name,
            text=text,
            tokens=tokens,
            meta={"subwords": len(token_strings), "masked": sum(1 for m in mask if not m)},
        )

    def explain_integrated_gradients(self, text: str) -> AttributionResult:
        """Integrated Gradients via Captum — detailed, on-demand."""
        try:
            from captum.attr import LayerIntegratedGradients
        except ImportError as e:
            raise XAIUnavailableError("captum is not installed (pip install captum)") from e

        model, tokenizer, model_name = self._load_classifier()
        torch = _import_torch()
        prepared = self._encode(tokenizer, text, self.settings.classifier_max_length)
        enc, ids = prepared["enc"], prepared["ids"]
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        target = injection_label_index(model.config)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        def forward(ids: Any, mask: Any) -> Any:
            return model(input_ids=ids, attention_mask=mask).logits

        lig = LayerIntegratedGradients(forward, model.get_input_embeddings())
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=torch.full_like(input_ids, pad_id),
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=self.settings.xai_ig_steps,
        )
        # attributions: [1, seq, emb_dim] -> per-token = L1 norm over embedding dim
        token_scores = attributions.sum(dim=-1)[0].abs()
        raw = [float(s) for s in token_scores.tolist()]
        token_strings: list[str] = tokenizer.convert_ids_to_tokens(ids)
        tokens = build_token_attribution(token_strings, raw, top_k=self.settings.xai_top_k)
        return AttributionResult(
            method="integrated_gradients",
            model=model_name,
            text=text,
            tokens=tokens,
            meta={
                "steps": self.settings.xai_ig_steps,
                "target": target,
                "subwords": len(token_strings),
            },
        )

    async def explain_text(self, text: str, method: str = "attention") -> AttributionResult:
        """Async wrapper — model work runs in a thread (debug/on-demand mode)."""
        if method == "attention":
            fn = self.explain_attention
        elif method == "integrated_gradients":
            fn = self.explain_integrated_gradients
        else:
            raise ValueError(f"unknown XAI method: {method}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, text)


def get_xai_manager(settings: Settings | None = None) -> XAIManager:
    return XAIManager(settings=settings)


# ---------------------------------------------------------------------------
# CLI — on-demand attribution for a single text (debug mode)
# ---------------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.xai",
        description="Explain which tokens trigger the prompt-injection classifier.",
    )
    parser.add_argument("text", help="text to attribute")
    parser.add_argument(
        "--method", choices=["attention", "integrated_gradients"], default="attention"
    )
    args = parser.parse_args()

    async def run() -> int:
        manager = get_xai_manager()
        try:
            result = await manager.explain_text(args.text, method=args.method)
        except XAIUnavailableError as e:
            print(f"XAI unavailable: {e}")
            return 1
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    return asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover - manual debug entrypoint
    raise SystemExit(_cli())
