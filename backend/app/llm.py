"""Thin Ollama LLM wrapper — streaming disabled, non-thinking mode enforced."""
from __future__ import annotations

import logging

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def generate(prompt: str, system: str = "") -> str:
    """Call Ollama and return the full text response."""
    payload: dict = {
        "model": settings.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.LLM_TEMPERATURE,
            "num_predict": settings.LLM_NUM_PREDICT,
        },
    }
    if system:
        payload["system"] = system

    # Qwen3 non-thinking mode: prepend /no_think to suppress <think> blocks
    if settings.LLM_THINK is False and "qwen" in settings.LLM_MODEL.lower():
        payload["prompt"] = "/no_think\n" + payload["prompt"]

    try:
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        text: str = resp.json().get("response", "")
        # Strip any residual <think>…</think> blocks
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except httpx.HTTPError as exc:
        logger.error("LLM call failed: %s", exc)
        raise


def is_ollama_up() -> bool:
    try:
        r = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False
