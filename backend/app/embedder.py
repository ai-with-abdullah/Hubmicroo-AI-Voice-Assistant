"""Singleton BGE-M3 embedder — loaded once, reused across requests."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from .config import get_settings

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> "BGEM3FlagModel":
    from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

    logger.info("Loading BGE-M3 model on device=%s …", settings.EMBED_DEVICE)
    model = BGEM3FlagModel(
        settings.EMBED_MODEL,
        use_fp16=(settings.EMBED_DEVICE != "cpu"),
    )
    logger.info("BGE-M3 ready")
    return model


def embed_text(text: str) -> list[float]:
    """Return the dense embedding vector for *text* (BGE-M3, dim=1024)."""
    model = _get_model()
    result = model.encode([text], batch_size=1, return_dense=True)
    return result["dense_vecs"][0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of texts."""
    if not texts:
        return []
    model = _get_model()
    result = model.encode(texts, batch_size=32, return_dense=True)
    return [v.tolist() for v in result["dense_vecs"]]
