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
    import torch  # noqa: PLC0415

    device = settings.EMBED_DEVICE
    # Auto-upgrade to CUDA when available and config was left at the default "cpu".
    # On Kaggle T4 this cuts per-call latency from ~3 s to ~50 ms.
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA detected — auto-upgrading EMBED_DEVICE cpu → cuda")

    logger.info("Loading BGE-M3 via sentence-transformers on device=%s …", device)
    model = SentenceTransformer(
        settings.EMBED_MODEL,
        device=device,
        trust_remote_code=True,
    )
    logger.info("BGE-M3 ready on %s", device)
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
