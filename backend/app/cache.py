"""Semantic cache: keyed by query embedding, served on cosine similarity hit.

Uses an in-process OrderedDict (LRU eviction) — no external cache service needed.
Thread-safe for asyncio (single-threaded event loop per process).
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .config import get_settings

settings = get_settings()


@dataclass
class _CacheEntry:
    answer: str
    products: list[dict[str, Any]]
    lang: str
    ts: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(
        self,
        threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        max_size: int = settings.CACHE_MAX_SIZE,
    ):
        self._threshold = threshold
        self._max_size = max_size
        # {vector_tuple: _CacheEntry}  — OrderedDict maintains insertion order
        self._store: OrderedDict[tuple, _CacheEntry] = OrderedDict()

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: list[float], b: tuple) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _evict(self) -> None:
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # remove oldest

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, vector: list[float]) -> _CacheEntry | None:
        """Return cached entry if a stored vector is within threshold."""
        best_sim = 0.0
        best_key: tuple | None = None
        for key in self._store:
            sim = self._cosine(vector, key)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key is not None and best_sim >= self._threshold:
            # Move to end (mark as recently used)
            self._store.move_to_end(best_key)
            return self._store[best_key]
        return None

    def set(
        self,
        vector: list[float],
        answer: str,
        products: list[dict[str, Any]],
        lang: str,
    ) -> None:
        key = tuple(vector)
        self._evict()
        self._store[key] = _CacheEntry(answer=answer, products=products, lang=lang)
        self._store.move_to_end(key)

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store), "max_size": self._max_size}


# Singleton — one cache per process
_cache = SemanticCache()


def get_cache() -> SemanticCache:
    return _cache
