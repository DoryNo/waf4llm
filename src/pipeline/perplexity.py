from __future__ import annotations

import math
import re
from collections import Counter
from contextlib import suppress
from typing import Protocol


class PerplexityScorer(Protocol):
    def score(self, text: str) -> float:
        """Return perplexity proxy confidence 0..1 (higher = more suspicious)."""
        ...


class HeuristicPerplexityScorer:
    """Lightweight heuristic fallback for GCG-style suffix detection without LM.

    Flags high-perplexity gibberish via:
    - punctuation/bracket density
    - character entropy
    - non-dictionary token ratio
    - repeated special tokens like '}{', '][', '\\', '|||'
    """

    name = "heuristic"

    # Common English + Russian words for quick dictionary check (small subset, enough to separate natural text from gibberish)
    COMMON_WORDS = {
        "the",
        "and",
        "you",
        "are",
        "for",
        "with",
        "this",
        "that",
        "have",
        "from",
        "not",
        "but",
        "what",
        "all",
        "can",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "hello",
        "world",
        "please",
        "help",
        "thanks",
        "today",
        "ignore",
        "instruction",
        "system",
        "prompt",
        "previous",
        "summarize",
        "capital",
        "france",
        "document",
        "context",
        # Russian common
        "и",
        "в",
        "не",
        "на",
        "я",
        "что",
        "тот",
        "быть",
        "с",
        "а",
        "весь",
        "это",
        "как",
        "она",
        "по",
        "но",
        "они",
        "к",
        "у",
        "ты",
        "привет",
        "мир",
        "пожалуйста",
        "спасибо",
        "сегодня",
        "игнорируй",
        "инструкции",
        "система",
        "промпт",
        "документ",
    }

    GCG_PATTERNS = [
        re.compile(r"[\}\{\]\[\\\|`~]{3,}"),  # brackets/backslashes run
        re.compile(r"(?:\b\w{1,3}\s*){10,}.*[\}\]\[\\]{2,}"),  # many short tokens + brackets
        re.compile(r"(?:[A-Za-z]{1,2}[\W_]){5,}"),  # alternating letter-punct like "a]b]c]d]"
    ]

    def _punct_density(self, text: str) -> float:
        if not text:
            return 0.0
        punct = sum(1 for c in text if c in "}]{[\\|/`~!@#$%^&*=_+;:<>?\"'")
        return punct / len(text)

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        c = Counter(text)
        length = len(text)
        return -sum((v / length) * math.log2(v / length) for v in c.values())

    def _nondict_ratio(self, text: str) -> float:
        words = re.findall(r"[A-Za-zА-Яа-я]{2,}", text.lower())
        if not words:
            return 0.0
        nondict = sum(1 for w in words if w not in self.COMMON_WORDS and len(w) > 3)
        return nondict / len(words)

    def score(self, text: str) -> float:
        if not text or len(text.strip()) < 20:
            return 0.0
        # Quick GCG pattern match — high confidence
        for pat in self.GCG_PATTERNS:
            if pat.search(text) and self._punct_density(text) > 0.12 and self._entropy(text) > 4.0:
                return 0.78

        punct = self._punct_density(text)
        ent = self._entropy(text)
        nondict = self._nondict_ratio(text)

        # Heuristic combination
        # Natural English: punct ~0.02-0.05, ent ~4.0-4.5, nondict ~0.3
        # GCG gibberish: punct >0.12, ent >4.0, nondict >0.5
        score = 0.0
        if punct > 0.10 and ent > 4.0 and nondict > 0.5:
            score = 0.75
        elif punct > 0.15 and ent > 4.2:
            score = 0.82
        elif ent > 4.8 and nondict > 0.65:
            score = 0.70
        elif punct > 0.30:
            # Very high punct density alone is suspicious (e.g., "}]{[\\|~")
            score = max(score, 0.65)
        # Check for adversarial suffix style: benign prefix + gibberish suffix
        if len(text) > 60:
            suffix = text[-50:]
            s_punct = self._punct_density(suffix)
            s_ent = self._entropy(suffix)
            if s_punct > 0.20 and s_ent > 4.0 and punct < 0.12:
                score = max(score, 0.80)
            # Also check overall suffix punct high even if overall punct moderate
            if s_punct > 0.30:
                score = max(score, 0.75)

        return score


class TransformersPerplexityScorer:
    """True perplexity scorer using small causal LM (distilgpt2/gpt2).

    Computes per-token perplexity: ppl = exp(avg NLL). Thresholds were tuned
    once on the frozen holdout distribution (documented in
    benchmarks/eval_dataset.md): benign English text (alpaca) stays mostly
    below ~500 ppl, while GCG/DSN adversarial suffixes sit above ~800. Short
    texts (< min_chars) are skipped: ppl estimates on short strings are too
    noisy and non-English snippets produce spuriously high values.

    Mapping: ppl > 8000 -> 0.92 (block), > 1500 -> 0.78 (exclude),
    > 800 -> 0.62 (sanitize); below -> 0.
    """

    name = "transformers"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_length: int = 512,
        device: str = "cpu",
        min_chars: int = 60,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.min_chars = min_chars
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._model is not None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=False)
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
            model.eval()
            if self.device != "cpu":
                with suppress(Exception):
                    model.to(self.device)
            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = True
            self._model = None
            return False

    def _compute_ppl(self, text: str) -> float | None:
        if not self._ensure_loaded() or self._model is None or self._tokenizer is None:
            return None
        try:
            import torch

            enc = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=self.max_length
            )
            input_ids = enc.input_ids
            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                # outputs.loss is avg NLL
                loss = outputs.loss
                ppl = math.exp(loss.item())
                return ppl
        except Exception:
            return None

    def score(self, text: str) -> float:
        if not text or len(text.strip()) < self.min_chars:
            return 0.0
        ppl = self._compute_ppl(text)
        if ppl is None:
            # Fallback to heuristic if model failed
            return HeuristicPerplexityScorer().score(text)
        # Map perplexity to confidence (tuned on the frozen holdout, see docstring)
        if ppl > 8000:
            return 0.92
        if ppl > 1500:
            return 0.78
        if ppl > 800:
            return 0.62
        return 0.0

    @property
    def load_error(self) -> str | None:
        return self._load_error


def get_perplexity_scorer(
    enabled: bool = True,
    model_name: str = "distilgpt2",
    use_transformers: bool = True,
    min_chars: int = 60,
) -> PerplexityScorer | None:
    if not enabled:
        return None
    if use_transformers:
        try:
            import transformers  # noqa: F401

            # Try transformers scorer; if load fails it will fallback internally
            return TransformersPerplexityScorer(model_name=model_name, min_chars=min_chars)
        except ImportError:
            pass
    return HeuristicPerplexityScorer()
