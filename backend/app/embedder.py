"""Singleton BGE-M3 embedder — uses sentence-transformers to avoid FlagEmbedding's
finetune import chain which triggers a torch.load version check (CVE-2025-32434).
Vectors are identical: 1024-dim cosine-normalised dense floats."""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    logger.info("Loading BGE-M3 via sentence-transformers on device=%s …", settings.EMBED_DEVICE)
    model = SentenceTransformer(
        settings.EMBED_MODEL,
        device=settings.EMBED_DEVICE,
        trust_remote_code=True,
    )
    logger.info("BGE-M3 ready")
    return model


def embed_text(text: str) -> list[float]:
    """Return the dense embedding vector for *text* (BGE-M3, dim=1024)."""
    model = _get_model()
    vecs = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vecs[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of texts."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vecs]
