"""Voice pipeline — Whisper STT and Piper TTS (self-hosted, no browser APIs).

All audio stays on the server. Raw audio is never persisted after transcription.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .config import get_settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Speech-to-Text (Whisper) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_whisper() -> "WhisperModel":
    from faster_whisper import WhisperModel  # noqa: PLC0415

    logger.info("Loading Whisper model '%s' on %s …", settings.WHISPER_MODEL, settings.WHISPER_DEVICE)
    model = WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type="int8" if settings.WHISPER_DEVICE == "cpu" else "float16",
    )
    logger.info("Whisper ready")
    return model


def transcribe(audio_bytes: bytes, hint_lang: str | None = None) -> tuple[str, str]:
    """Transcribe audio bytes; return (transcript, detected_language).

    *hint_lang* is 'en'|'ur'|'ar' — passed as initial_prompt when provided.
    """
    model = _get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segments, info = model.transcribe(
            tmp_path,
            language=hint_lang if hint_lang in ("en", "ar") else None,
            initial_prompt="Urdu or English shopping assistant" if hint_lang == "ur" else None,
            vad_filter=True,
        )
        text = " ".join(s.text for s in segments).strip()
        detected = info.language or hint_lang or "en"
        return text, detected
    finally:
        os.unlink(tmp_path)  # never persist audio on disk


# ── Text-to-Speech (Piper) ────────────────────────────────────────────────────

_PIPER_VOICE_MAP = {
    "en": settings.PIPER_VOICE_EN,
    "ur": settings.PIPER_VOICE_UR,
    "ar": settings.PIPER_VOICE_AR,
}


def synthesize(text: str, lang: str) -> bytes | None:
    """Convert text to WAV bytes using Piper. Returns None if TTS is disabled."""
    if not settings.TTS_ENABLED:
        return None

    voice = _PIPER_VOICE_MAP.get(lang, settings.PIPER_VOICE_EN)
    voice_path = Path(settings.PIPER_VOICES_DIR) / f"{voice}.onnx"

    if not voice_path.exists():
        logger.warning("Piper voice model not found: %s", voice_path)
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = out_f.name

    try:
        result = subprocess.run(
            ["piper", "--model", str(voice_path), "--output_file", out_path],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error("Piper failed: %s", result.stderr.decode())
            return None
        return Path(out_path).read_bytes()
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error("Piper TTS error: %s", exc)
        return None
    finally:
        if Path(out_path).exists():
            os.unlink(out_path)
