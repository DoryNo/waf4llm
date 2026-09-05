from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.pipeline.base import (
    DecisionLevel,
    LayerResult,
    PipelineContext,
    ProvenanceTag,
    decision_level_for_confidence,
)
from src.pipeline.perplexity import (
    HeuristicPerplexityScorer,
    PerplexityScorer,
    get_perplexity_scorer,
)

logger = get_logger("heuristic")


class HeuristicLayer:
    """Fast regex-based filter + perplexity scorer for known injection patterns."""

    name = "heuristic"

    # Fallback hardcoded patterns if YAML not found (mirrors heuristic.yaml)
    FALLBACK_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
        (
            re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
            0.85,
            "ignore-previous-instructions",
        ),
        (
            re.compile(r"disregard\s+(all\s+)?(previous|above|prior)", re.I),
            0.80,
            "disregard-previous",
        ),
        (
            re.compile(r"you\s+are\s+now\s+(a\s+)?(dan|jailbreak|unrestricted|unfiltered)", re.I),
            0.90,
            "role-hijack-dan",
        ),
        (re.compile(r"system\s*:\s*you\s+are", re.I), 0.70, "fake-system-prompt"),
        (re.compile(r"\[SYSTEM\]|\[INST\]|\<\|\s*im_start\s*\|>"), 0.75, "fake-chat-markers"),
        (
            re.compile(r"reveal\s+(your\s+)?(system|initial|original)\s+prompt", re.I),
            0.85,
            "prompt-extraction",
        ),
        (
            re.compile(r"repeat\s+(your\s+)?(system|initial)\s+(prompt|instructions)", re.I),
            0.80,
            "prompt-extraction-repeat",
        ),
        (re.compile(r"do\s+anything\s+now", re.I), 0.75, "dan-trigger"),
        (
            re.compile(r"exfiltrate|leak\s+(secrets?|keys?|tokens?)", re.I),
            0.70,
            "exfiltration-intent",
        ),
        (
            re.compile(r"execute\s+(the\s+)?(following\s+)?(code|command|tool)", re.I),
            0.60,
            "code-execution-request",
        ),
        (re.compile(r"bypass\s+(safety|filter|guardrail|policy)", re.I), 0.80, "bypass-request"),
        (re.compile(r"jailbreak|jail\s*break", re.I), 0.65, "jailbreak-keyword"),
        (re.compile(r"developer\s+mode|god\s+mode", re.I), 0.70, "privileged-mode-request"),
        (
            re.compile(
                r"игнорируй\s+(все\s+)?(предыдущие|прошлые|системные)\s+(инструкции|указания|промпты)",
                re.I,
            ),
            0.85,
            "ignore-previous-instructions-ru",
        ),
        (
            re.compile(r"забудь\s+(все\s+)?(предыдущие|прошлые)\s+(инструкции|указания)", re.I),
            0.80,
            "forget-previous-ru",
        ),
        (
            re.compile(r"ты\s+теперь\s+(dan|джейлбрейк|без\s+ограничений)", re.I),
            0.90,
            "role-hijack-dan-ru",
        ),
        (
            re.compile(r"раскрой\s+(свой\s+)?(системный|начальный|исходный)\s+промпт", re.I),
            0.85,
            "prompt-extraction-ru",
        ),
        (
            re.compile(r"обойди\s+(защиту|фильтр|модерацию|политику|ограничения)", re.I),
            0.80,
            "bypass-request-ru",
        ),
        (re.compile(r"джейлбрейк", re.I), 0.65, "jailbreak-keyword-ru"),
        (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I), 0.62, "roleplay-pretend"),
        (re.compile(r"притворись\s+(что\s+ты|кем)", re.I), 0.62, "roleplay-pretend-ru"),
        (re.compile(r"from\s+now\s+on\s+you\s+(are|will)", re.I), 0.66, "from-now-on"),
        (re.compile(r"отныне\s+ты\s+", re.I), 0.66, "from-now-on-ru"),
    ]

    RET_WEIGHT_MULTIPLIER = 0.4
    TOOL_WEIGHT_MULTIPLIER = 0.5

    def __init__(
        self,
        rules_path: str | None = None,
        perplexity_enabled: bool | None = None,
        perplexity_model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rules_path = Path(rules_path or self.settings.heuristic_rules_path)
        self.perplexity_enabled = (
            perplexity_enabled
            if perplexity_enabled is not None
            else self.settings.heuristic_perplexity_enabled
        )
        self.perplexity_model = perplexity_model or self.settings.heuristic_perplexity_model

        self._patterns: list[tuple[re.Pattern[str], float, str]] = list(self.FALLBACK_PATTERNS)
        self._rules_mtime: float | None = None
        self._rules_load_error: str | None = None
        self._perplexity_scorer: PerplexityScorer | None = None
        self._perplexity_init_tried = False

        # Initial load
        self._load_rules_if_needed(force=True)

    def _load_rules_if_needed(self, force: bool = False) -> None:
        try:
            if not self.rules_path.exists():
                if force:
                    logger.warning(
                        "heuristic rules file not found, using fallback", path=str(self.rules_path)
                    )
                return
            mtime = self.rules_path.stat().st_mtime
            if not force and self._rules_mtime is not None and mtime <= self._rules_mtime:
                return
            # Load YAML
            data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
            rules = data.get("rules") or []
            patterns: list[tuple[re.Pattern[str], float, str]] = []
            for r in rules:
                try:
                    pattern = r.get("pattern")
                    weight = float(r.get("weight", 0.5))
                    label = r.get("label", pattern[:30])
                    if not pattern:
                        continue
                    # Compile with case-insensitive by default; allow inline flags
                    compiled = re.compile(pattern, re.I)
                    patterns.append((compiled, weight, label))
                except Exception as e:
                    logger.warning(
                        "invalid heuristic rule skipped", pattern=r.get("pattern"), error=str(e)
                    )
            if patterns:
                self._patterns = patterns
                self._rules_mtime = mtime
                self._rules_load_error = None
                logger.info(
                    "heuristic rules loaded", path=str(self.rules_path), count=len(patterns)
                )
            else:
                logger.warning(
                    "heuristic rules empty, keeping previous patterns", path=str(self.rules_path)
                )
        except Exception as e:
            self._rules_load_error = str(e)
            logger.error("failed to load heuristic rules", path=str(self.rules_path), error=str(e))

    def _get_perplexity_scorer(self):
        if not self.perplexity_enabled:
            return None
        if self._perplexity_scorer is not None:
            return self._perplexity_scorer
        if self._perplexity_init_tried:
            return self._perplexity_scorer
        self._perplexity_init_tried = True
        try:
            # Try transformers scorer, fallback to heuristic if unavailable
            scorer = get_perplexity_scorer(
                enabled=True,
                model_name=self.perplexity_model,
                use_transformers=True,
                min_chars=self.settings.heuristic_perplexity_min_chars,
            )
            self._perplexity_scorer = scorer
            if scorer and hasattr(scorer, "name"):
                logger.info(
                    "perplexity scorer initialized", scorer=scorer.name, model=self.perplexity_model
                )
            return scorer
        except Exception as e:
            logger.warning("perplexity scorer init failed, using heuristic fallback", error=str(e))
            self._perplexity_scorer = HeuristicPerplexityScorer()
            return self._perplexity_scorer

    def reload_rules(self) -> None:
        self._load_rules_if_needed(force=True)

    @property
    def patterns(self) -> list[tuple[re.Pattern[str], float, str]]:
        # Hot-reload check on each access (cheap mtime stat)
        self._load_rules_if_needed(force=False)
        return self._patterns

    def _score_regex(self, text: str, tag: ProvenanceTag) -> tuple[float, list[str]]:
        max_score = 0.0
        triggers: list[str] = []
        multiplier = 1.0
        if tag == ProvenanceTag.RET:
            multiplier = self.RET_WEIGHT_MULTIPLIER
        elif tag == ProvenanceTag.TOOL:
            multiplier = self.TOOL_WEIGHT_MULTIPLIER

        for pattern, weight, label in self.patterns:
            try:
                if pattern.search(text):
                    score = weight * multiplier
                    triggers.append(label)
                    if score > max_score:
                        max_score = score
            except Exception:
                continue
        return max_score, triggers

    def _score_perplexity(self, text: str, tag: ProvenanceTag) -> tuple[float, list[str]]:
        scorer = self._get_perplexity_scorer()
        if scorer is None:
            return 0.0, []
        try:
            # Perplexity is more suspicious for USR than for RET (RET may contain code/gibberish legitimately)
            # But GCG suffixes are typically in USR
            # Apply multiplier similar to regex, but less aggressive for RET
            conf = scorer.score(text)
            if conf == 0.0:
                return 0.0, []
            multiplier = 1.0
            if tag == ProvenanceTag.RET:
                multiplier = 0.6
            elif tag == ProvenanceTag.TOOL:
                multiplier = 0.5
            adjusted = conf * multiplier
            if adjusted > 0.3:
                return adjusted, ["high-perplexity-gcg"]
            return 0.0, []
        except Exception as e:
            logger.warning("perplexity scoring failed", error=str(e))
            return 0.0, []

    def _score_segment(self, text: str, tag: ProvenanceTag) -> tuple[float, list[str]]:
        # Combine regex + perplexity via max (graduated response picks highest)
        regex_score, regex_triggers = self._score_regex(text, tag)
        ppl_score, ppl_triggers = self._score_perplexity(text, tag)

        # Combine triggers
        all_triggers = list(regex_triggers)
        if ppl_triggers:
            all_triggers.extend(ppl_triggers)

        # Final score is max, but if both moderate, boost slightly
        combined = max(regex_score, ppl_score)
        if regex_score > 0.4 and ppl_score > 0.5:
            combined = max(combined, min(0.85, (regex_score + ppl_score) / 2 + 0.1))

        return combined, all_triggers

    async def process(self, ctx: PipelineContext) -> LayerResult:
        if not ctx.segments:
            return LayerResult(layer=self.name, passed=True)

        # Hot-reload rules check
        self._load_rules_if_needed(force=False)

        best_score = 0.0
        all_triggers: list[str] = []
        flagged_segment_idx: int | None = None
        per_segment_scores: list[dict[str, Any]] = []

        for seg in ctx.segments:
            # Check both normalized and original content for robustness against homoglyph mangling
            # (normalizer maps Cyrillic lookalikes to Latin for detection of obfuscated EN,
            # but that breaks RU detection if only normalized is checked)
            eff = seg.effective_content()
            orig = seg.content
            score_eff, trig_eff = self._score_segment(eff, seg.tag)
            # If original differs from effective, also score original and take max
            if orig != eff and orig.strip():
                score_orig, trig_orig = self._score_segment(orig, seg.tag)
                # Merge: take max score, union triggers
                if score_orig > score_eff:
                    score, triggers = score_orig, trig_orig
                    # Include eff triggers as well for observability if they exist
                    if trig_eff:
                        for t in trig_eff:
                            if t not in triggers:
                                triggers.append(t)
                else:
                    score, triggers = score_eff, trig_eff
                    if trig_orig and score_orig > 0.3:
                        for t in trig_orig:
                            if t not in triggers:
                                triggers.append(t)
            else:
                score, triggers = score_eff, trig_eff
            per_segment_scores.append(
                {"idx": seg.index, "tag": seg.tag.value, "score": score, "triggers": triggers}
            )
            if triggers:
                # Deduplicate triggers globally but keep per-segment
                for t in triggers:
                    if t not in all_triggers:
                        all_triggers.append(t)
            if score > best_score:
                best_score = score
                flagged_segment_idx = seg.index

        if best_score == 0.0:
            return LayerResult(
                layer=self.name,
                passed=True,
                confidence=0.0,
                extra={"per_segment": per_segment_scores},
            )

        level = decision_level_for_confidence(
            best_score,
            threshold_log=self.settings.threshold_log,
            threshold_sanitize=self.settings.threshold_sanitize,
            threshold_exclude=self.settings.threshold_exclude,
            threshold_block=self.settings.threshold_block,
        )

        passed = level in (DecisionLevel.allow, DecisionLevel.log)

        return LayerResult(
            layer=self.name,
            passed=passed,
            confidence=best_score,
            level=level,
            reason=f"heuristic match: {', '.join(all_triggers)}"
            if all_triggers
            else "perplexity anomaly",
            trigger_tokens=all_triggers,
            extra={
                "flagged_segment": flagged_segment_idx,
                "per_segment": per_segment_scores,
                "rules_mtime": self._rules_mtime,
            },
        )
