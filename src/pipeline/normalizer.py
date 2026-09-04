from __future__ import annotations

import base64
import codecs
import math
import re
import unicodedata
from collections import Counter

from src.pipeline.base import LayerResult, PipelineContext


class NormalizerLayer:
    """Anti-evasion normalization per Phase 3 spec.

    Steps (in order):
    1. NFKC normalization
    2. Zero-width / invisible char removal (expanded set)
    3. Homoglyph mapping (Cyrillic/Greek → Latin) for detection
    4. Sparse text de-obfuscation: "i g n o r e" → "ignore"
    5. Recursive decoding: Base64 / Hex / ROT13 / URL with depth limit + printable gate + entropy pre-filter + bomb limits
    6. Leet-speak normalization (word-level, not numeric)
    """

    name = "normalizer"

    MAX_DECODE_DEPTH = 3
    MIN_B64_LEN = 8
    MAX_DECODED_SIZE = 100_000
    MAX_INPUT_SIZE = 500_000
    MAX_EXPANSION_FACTOR = 10

    # Expanded zero-width / invisible characters
    # Includes ZWSP, ZWNJ, ZWJ, BOM, soft hyphen, word joiner, mongolian vowel separator,
    # LRM/RLM, isolate/embedding controls, hangul fillers
    ZERO_WIDTH_RE = re.compile(
        r"[\u200b-\u200d\ufeff\u00ad\u2060\u180e\u200e\u200f\u2061-\u2064\u115f\u1160\u3164\u202a-\u202e\u2066-\u2069]"
    )

    # Sparse text: "h e l l o" -> "hello" — sequence of single-char tokens separated by spaces
    SPARSE_RE = re.compile(r"\b(?:\w\s+){3,}\w\b")

    # Homoglyph mapping: Cyrillic / Greek lookalikes → Latin (for detection in normalized_content)
    # Note: applied unconditionally in normalized version for injection detection; original preserved.
    HOMOGLYPH_MAP = str.maketrans(
        {
            # Cyrillic lower
            "\u0430": "a",  # а
            "\u0435": "e",  # е
            "\u043e": "o",  # о
            "\u0440": "p",  # р
            "\u0441": "c",  # с
            "\u0443": "y",  # у
            "\u0445": "x",  # х
            "\u0456": "i",  # і (Ukrainian)
            "\u0454": "e",  # є
            "\u0457": "i",  # ї
            # Cyrillic upper
            "\u0410": "A",  # А
            "\u0412": "B",  # В
            "\u0415": "E",  # Е
            "\u041a": "K",  # К
            "\u041c": "M",  # М
            "\u041d": "H",  # Н
            "\u041e": "O",  # О
            "\u0420": "P",  # Р
            "\u0421": "C",  # С
            "\u0422": "T",  # Т
            "\u0425": "X",  # Х
            # Greek lower
            "\u03b1": "a",  # α
            "\u03b5": "e",  # ε
            "\u03bf": "o",  # ο
            "\u03c1": "p",  # ρ
            "\u03c2": "s",  # ς
            "\u03c3": "s",  # σ
            "\u03c5": "y",  # υ
            "\u03c7": "x",  # χ
            "\u03b9": "i",  # ι
            "\u03ba": "k",  # κ
            "\u03bd": "v",  # ν (looks like v)
            # Greek upper
            "\u0391": "A",
            "\u0395": "E",
            "\u039f": "O",
            "\u03a1": "P",
            "\u03a3": "S",
            "\u03a7": "X",
            "\u039a": "K",
            "\u039c": "M",
            "\u039d": "N",
            "\u03a4": "T",
            # Additional confusables
            "\u04cf": "b",  # ӏ (Cyrillic palochka → l/b)
            "\u043a": "k",  # к (Cyrillic small ka)
            "\u043d": "h",  # н (en) resembles h? but mapping to h for detection
        }
    )

    LEET_MAP = str.maketrans(
        {
            "0": "o",
            "1": "i",
            "3": "e",
            "4": "a",
            "5": "s",
            "6": "g",
            "7": "t",
            "8": "b",
            "@": "a",
            "$": "s",
            "!": "i",
        }
    )

    # Regex for encoded blobs — allow short payloads (8 chars for b64, 12 hex chars for hex) to catch short injection keywords
    B64_RE = re.compile(r"(?:[A-Za-z0-9+/]{8,}={0,2})")
    # Hex: at least 12 chars (6 bytes) to catch "ignore" (12 hex chars) while still avoiding 2-char false positives
    HEX_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}){6,}\b")
    URL_ENCODED_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){4,}")

    # Injection keywords for leet/rot13 heuristics (also used to gate translations)
    INJECTION_KEYWORDS = (
        "ignore",
        "instruction",
        "system",
        "jailbreak",
        "prompt",
        "previous",
        "disregard",
        "bypass",
        "dan",
        "roleplay",
        "unrestricted",
    )

    @staticmethod
    def shannon_entropy(s: str) -> float:
        if not s:
            return 0.0
        counts = Counter(s)
        length = len(s)
        return -sum((c / length) * math.log2(c / length) for c in counts.values())

    def _is_mostly_printable(self, s: str, threshold: float = 0.85) -> bool:
        if not s:
            return False
        if len(s) > self.MAX_DECODED_SIZE:
            return False
        printable = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
        return (printable / len(s)) >= threshold

    def _normalize_homoglyphs(self, text: str) -> str:
        """Map lookalikes only in mixed-script tokens.

        Mapping every Cyrillic character in a Russian sentence would corrupt normal
        Russian input. Mixed-script tokens are the useful signal for Latin payloads
        obfuscated with Cyrillic/Greek characters.
        """

        def replace_token(match: re.Match[str]) -> str:
            token = match.group(0)
            has_latin = any("LATIN" in unicodedata.name(char, "") for char in token)
            has_cyrillic = any("CYRILLIC" in unicodedata.name(char, "") for char in token)
            has_greek = any("GREEK" in unicodedata.name(char, "") for char in token)

            # Preserve native Cyrillic words; map mixed Cyrillic/Latin tokens and Greek tokens.
            if has_cyrillic and not has_latin:
                return token
            if has_latin or has_greek:
                return token.translate(self.HOMOGLYPH_MAP)
            return token

        return re.sub(r"\w+", replace_token, text, flags=re.UNICODE)

    def _entropy_gate_b64(self, candidate: str) -> bool:
        if len(candidate) < self.MIN_B64_LEN:
            return False
        # Exclude low-entropy repeated patterns like "AAAAAAAAAAAAAAAA"
        ent = self.shannon_entropy(candidate)
        # Dynamic threshold: short candidates (8-12 chars) have naturally lower max entropy (log2(n))
        # Use 2.5 for short, 3.0 for longer
        threshold = 2.5 if len(candidate) < 12 else 3.0
        if ent < threshold:
            return False
        # Reject strings with very low diversity (single char repeated or 2-char pattern)
        return not len(set(candidate.rstrip("="))) < 4

    def _entropy_gate_hex(self, candidate: str) -> bool:
        cleaned = re.sub(r"\s+", "", candidate)
        if len(cleaned) < 12 or len(cleaned) % 2 != 0:  # 12 chars = 6 bytes, enough for "ignore"
            return False
        # Hex should be only hex chars, entropy moderate (2.0-4.5) — lower bound relaxed to allow low-entropy hex like "69676e..."
        ent = self.shannon_entropy(cleaned.lower())
        if ent < 2.0 or ent > 4.5:
            return False
        # Reject strings that are mostly repeated (e.g., "000000...")
        return not len(set(cleaned.lower())) < 4

    def _try_b64_decode(self, s: str) -> str | None:
        cleaned = s.strip()
        if not self._entropy_gate_b64(cleaned):
            return None
        padded = cleaned + "=" * (-len(cleaned) % 4)
        try:
            decoded = base64.b64decode(padded, validate=True)
            # Bomb limit: decoded size
            if (
                len(decoded) > self.MAX_DECODED_SIZE
                or len(decoded) > len(cleaned) * self.MAX_EXPANSION_FACTOR
            ):
                return None
            text = decoded.decode("utf-8", errors="strict")
            if self._is_mostly_printable(text) and len(text) >= 4:
                return text
        except Exception:
            pass
        return None

    def _try_hex_decode(self, s: str) -> str | None:
        if not self._entropy_gate_hex(s):
            return None
        cleaned = re.sub(r"\s+", "", s.strip())
        if len(cleaned) % 2 != 0:
            return None
        try:
            decoded = bytes.fromhex(cleaned)
            if (
                len(decoded) > self.MAX_DECODED_SIZE
                or len(decoded) > len(cleaned) * self.MAX_EXPANSION_FACTOR
            ):
                return None
            text = decoded.decode("utf-8", errors="strict")
            if self._is_mostly_printable(text) and len(text) >= 4:
                return text
        except Exception:
            pass
        return None

    def _try_url_decode(self, s: str) -> str | None:
        # Look for %XX sequences
        if "%" not in s:
            return None
        # Only attempt if multiple % encodings present
        if s.count("%") < 2:
            return None
        try:
            import urllib.parse

            decoded = urllib.parse.unquote(s, encoding="utf-8", errors="strict")
            if decoded != s and self._is_mostly_printable(decoded) and len(decoded) >= 4:
                # Check if decoded contains injection-like keywords for confidence
                if any(
                    kw in decoded.lower()
                    for kw in ("ignore", "instruction", "system", "jailbreak", "prompt", "previous")
                ):
                    return decoded
                # Even without keywords, if decoded is plausible English, return it
                # But to avoid over-decoding legitimate URLs, require that original looked encoded
                if self.URL_ENCODED_RE.search(s):
                    return decoded
        except Exception:
            pass
        return None

    def _try_rot13(self, s: str) -> str | None:
        # Only apply ROT13 if result looks like injection
        if len(s) < 8:
            return None
        # Quick check: ROT13 text often has same length/composition; heuristic to avoid false positives
        # Only decode if original contains no spaces that are already English-like injection?
        try:
            rot = codecs.decode(s, "rot_13")
        except Exception:
            return None
        if rot == s or not self._is_mostly_printable(rot):
            return None
        lower = rot.lower()
        # Require injection keyword in ROT13 decoded version but not in original (to avoid decoding normal text)
        injection_kws = (
            "ignore",
            "instruction",
            "system",
            "jailbreak",
            "prompt",
            "previous",
            "disregard",
            "bypass",
            "dan",
        )
        if any(kw in lower for kw in injection_kws):
            orig_lower = s.lower()
            if not any(kw in orig_lower for kw in injection_kws):
                return rot
        return None

    def _recursive_decode(self, text: str) -> str:
        if len(text) > self.MAX_INPUT_SIZE:
            # Truncate for performance, but keep original for detection; we decode first MAX_INPUT_SIZE chars
            text = text[: self.MAX_INPUT_SIZE]

        current = text
        for _depth in range(self.MAX_DECODE_DEPTH):
            decoded_this_round = None

            # Try URL decode on full text first (often outermost)
            url_candidate = self._try_url_decode(current)
            if url_candidate is not None:
                if (
                    len(url_candidate) > self.MAX_DECODED_SIZE
                    or len(url_candidate) > len(current) * self.MAX_EXPANSION_FACTOR
                ):
                    pass
                else:
                    current = url_candidate
                    decoded_this_round = url_candidate
                    continue

            # Try base64 — scan for b64 substrings, decode the first plausible one per round
            # We iterate through matches and try each
            for m in self.B64_RE.finditer(current):
                candidate = m.group(0)
                # Skip candidates that are part of larger word with spaces? Already by regex
                result = self._try_b64_decode(candidate)
                if result is not None:
                    # Bomb check: replacement wouldn't explode size
                    if len(current) - len(candidate) + len(result) > self.MAX_DECODED_SIZE:
                        continue
                    current = current[: m.start()] + result + current[m.end() :]
                    decoded_this_round = result
                    break
            if decoded_this_round is not None:
                continue

            # Try hex
            for m in self.HEX_RE.finditer(current):
                candidate = m.group(0)
                result = self._try_hex_decode(candidate)
                if result is not None:
                    if len(current) - len(candidate) + len(result) > self.MAX_DECODED_SIZE:
                        continue
                    current = current[: m.start()] + result + current[m.end() :]
                    decoded_this_round = result
                    break
            if decoded_this_round is not None:
                continue

            # Try ROT13 on full text (only if not already decoded something)
            rot = self._try_rot13(current)
            if rot is not None:
                if (
                    len(rot) > self.MAX_DECODED_SIZE
                    or len(rot) > len(current) * self.MAX_EXPANSION_FACTOR
                ):
                    pass
                else:
                    current = rot
                    continue

            break
        return current

    def _should_translate_leet_word(self, word: str) -> bool:
        if not word:
            return False
        stripped = word.strip(".,;:!?()[]{}\"'`-_")
        if not stripped:
            return False
        # Pure numeric (with separators) -> skip (protects 2024, UUID parts, hashes)
        cleaned_for_digit = (
            stripped.replace("-", "")
            .replace("_", "")
            .replace(".", "")
            .replace("/", "")
            .replace(":", "")
        )
        if cleaned_for_digit.isdigit():
            return False
        # Must contain at least one leet symbol and at least one letter
        has_leet_symbol = any(c in "01345678@$!" for c in stripped)
        has_letter = any(c.isalpha() for c in stripped)
        if not (has_leet_symbol and has_letter):
            return False
        # Only translate if result contains known injection keyword — prevents mangling UUIDs/hashes/IDs
        # e.g., "550e8400" -> "ssoebaoo" does not contain injection keyword → skip
        # "1gn0r3" -> "ignore" contains "ignore" → translate
        translated = stripped.translate(self.LEET_MAP).lower()
        return any(kw in translated for kw in self.INJECTION_KEYWORDS)

    def _normalize_leetspeak(self, text: str) -> str:
        # Word-level leet normalization: only translate words that look leeted
        # Preserve whitespace/punctuation structure by splitting and rejoining via regex
        # Use regex to identify word tokens
        def repl_word(m: re.Match[str]) -> str:
            word = m.group(0)
            if self._should_translate_leet_word(word):
                return word.translate(self.LEET_MAP)
            return word

        # Pattern matches words containing alphanumerics and leet symbols
        return re.sub(r"[A-Za-z0-9@$!]+", repl_word, text)

    def normalize_text(self, text: str) -> str:
        if not text:
            return text
        # Truncate huge inputs for performance
        if len(text) > self.MAX_INPUT_SIZE:
            text = text[: self.MAX_INPUT_SIZE]

        # 1. NFKC normalization
        t = unicodedata.normalize("NFKC", text)
        # 2. Zero-width / invisible removal
        t = self.ZERO_WIDTH_RE.sub("", t)
        # 3. Homoglyph mapping (Cyrillic/Greek → Latin) for detection
        t = self._normalize_homoglyphs(t)

        # 4. Sparse text de-obfuscation: "i g n o r e" -> "ignore"
        def _desparse(m: re.Match[str]) -> str:
            return m.group(0).replace(" ", "")

        # Only apply if text looks suspiciously spaced: many single-char tokens
        if len(re.findall(r"\b\w\b", t)) >= 6:
            t = self.SPARSE_RE.sub(_desparse, t)

        # 5. Recursive decoding (b64/hex/url/rot13) with entropy gates and bomb limits
        t = self._recursive_decode(t)

        # Re-apply zero-width removal after decoding (decoded payload may contain obfuscation)
        t = self.ZERO_WIDTH_RE.sub("", t)

        # 6. Leet-speak normalization (word-level)
        t = self._normalize_leetspeak(t)

        return t

    async def process(self, ctx: PipelineContext) -> LayerResult:
        if not ctx.segments:
            return LayerResult(layer=self.name, passed=True, confidence=0.0, reason="no segments")

        for seg in ctx.segments:
            # Preserve original.content, set normalized_content for detection
            seg.normalized_content = self.normalize_text(seg.content)

        modified = sum(1 for s in ctx.segments if s.normalized_content != s.content)
        ctx.normalized = True
        return LayerResult(
            layer=self.name,
            passed=True,
            confidence=0.0,
            reason=f"normalized {modified}/{len(ctx.segments)} segments",
            extra={"modified_count": modified},
        )
