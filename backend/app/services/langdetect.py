"""Language detection for English / Urdu / Arabic.

Urdu and Arabic share the Arabic script, so a plain "is it Arabic script?"
test cannot tell them apart. The reliable signal is that Urdu uses a set of
extra letters that do not exist in Arabic (retroflex and Persian-derived
letters). We detect the script first, then disambiguate UR vs AR by looking
for those Urdu-only characters. English is detected by Latin letters.

This is pure-Python, instant, and needs no model or network call.
"""

# Letters that appear in Urdu but never in standard Arabic.
_URDU_ONLY = set("ٹڈڑںہھگچپژکیے")  # tta, ddal, rra, noon-ghunna, gaf, che, pe, etc.
# Core Arabic-script range.
_ARABIC_RANGE = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]


def _in_arabic_script(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_RANGE)


def detect_language(text: str) -> str:
    """Return 'en', 'ur', or 'ar'. Defaults to 'en' for empty/Latin text."""
    if not text or not text.strip():
        return "en"

    arabic_script = sum(1 for ch in text if _in_arabic_script(ch))
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    # Mostly Latin letters -> English.
    if arabic_script == 0 or latin > arabic_script:
        return "en"

    # Arabic script present: Urdu if it contains any Urdu-only letter.
    if any(ch in _URDU_ONLY for ch in text):
        return "ur"
    return "ar"
