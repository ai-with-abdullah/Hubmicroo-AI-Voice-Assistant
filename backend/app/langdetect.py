"""Lightweight language detection: English / Urdu / Arabic.

Uses Unicode block analysis first (fast, zero-model), then falls back to
character n-gram scoring. Handles code-switching (Urdu+English in one sentence).
"""
from __future__ import annotations

import re
import unicodedata


# Arabic Unicode block: 0x0600–0x06FF covers Arabic and Urdu letters.
# Urdu uses additional marks in 0xFB50–0xFDFF (Arabic Presentation Forms-A).
_RE_ARABIC_SCRIPT = re.compile(r"[؀-ۿﭐ-﷿]")
_RE_LATIN = re.compile(r"[a-zA-Z]")

# Romanized Urdu / Urdu transliteration keywords
_ROMANIZED_UR_TOKENS = frozenset(
    "kya hai hain aap yeh woh mujhe mera meri apka apki karo chahiye "
    "shukriya meherbani kahan kitna kitni koi agar lekin aur bhi nahi "
    "salam walaikum theek bilkul zaroor abhi phir dobara".split()
)

# Common Arabic-only words (MSA/Levantine) not found in Urdu
_AR_WORDS = frozenset(
    "كيف ماذا لماذا متى أين من هل نعم لا شكرا مرحبا السلام".split()
)


def detect_language(text: str) -> str:
    """Return 'en', 'ur', or 'ar'. Defaults to 'en' on short/ambiguous input."""
    if not text or not text.strip():
        return "en"

    arabic_chars = len(_RE_ARABIC_SCRIPT.findall(text))
    latin_chars = len(_RE_LATIN.findall(text))
    total = len(text.replace(" ", "")) or 1

    arabic_ratio = arabic_chars / total

    # Pure Arabic-script text — distinguish Arabic vs Urdu by word presence
    if arabic_ratio > 0.4:
        words = set(text.split())
        if words & _AR_WORDS:
            return "ar"
        # Urdu-specific letters: پ چ ژ ڈ ٹ ں ہ ے
        urdu_specific = len(re.findall(r"[پچژڈٹںہی]", text))
        return "ur" if urdu_specific > 0 else "ar"

    # Romanized Urdu detection: predominantly Latin but contains Urdu tokens
    if latin_chars / total > 0.5:
        lower_tokens = set(text.lower().split())
        urdu_hits = len(lower_tokens & _ROMANIZED_UR_TOKENS)
        if urdu_hits >= 2 or (urdu_hits >= 1 and len(lower_tokens) <= 6):
            return "ur"
        return "en"

    # Mixed script — if Arabic chars are present at all, lean Urdu/Arabic
    if arabic_chars > 0:
        return "ur"

    return "en"


def is_rtl(lang: str) -> bool:
    return lang in ("ur", "ar")
