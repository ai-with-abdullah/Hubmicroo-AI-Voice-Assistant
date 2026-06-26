"""Hybrid retrieval: dense (BGE-M3 via Qdrant) + sparse (BM25 via rank-bm25).

Scores are fused using weighted reciprocal rank fusion, then thresholded.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import get_settings
from .embedder import embed_text
from . import store

logger = logging.getLogger(__name__)
settings = get_settings()

_BM25_CORPUS_FILE = Path(settings.DATA_DIR) / "bm25_corpus.json"

# ── BM25 index (rebuilt when corpus changes) ──────────────────────────────────

_bm25_index: Any = None
_bm25_doc_ids: list[str] = []
_bm25_corpus_mtime: float = 0.0


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _get_bm25():
    """Lazy-load or refresh BM25 index when corpus file changes."""
    global _bm25_index, _bm25_doc_ids, _bm25_corpus_mtime

    if not _BM25_CORPUS_FILE.exists():
        return None, []

    mtime = _BM25_CORPUS_FILE.stat().st_mtime
    if _bm25_index is not None and mtime == _bm25_corpus_mtime:
        return _bm25_index, _bm25_doc_ids

    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    corpus_map: dict[str, str] = json.loads(_BM25_CORPUS_FILE.read_text())
    _bm25_doc_ids = list(corpus_map.keys())
    tokenized = [_tokenize(v) for v in corpus_map.values()]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus_mtime = mtime
    logger.debug("BM25 index refreshed: %d docs", len(_bm25_doc_ids))
    return _bm25_index, _bm25_doc_ids


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieve: returns ranked list of {score, payload} dicts.

    *doc_type* can be 'product', 'page', or None (both).
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    query_vec = embed_text(query)

    # ── Dense retrieval ────────────────────────────────────────────────
    dense_hits = store.search(query_vec, top_k=k * 2, doc_type=doc_type)
    dense_map: dict[str, float] = {
        h["payload"]["_doc_id"]: h["score"] for h in dense_hits
    }

    # ── BM25 retrieval ─────────────────────────────────────────────────
    bm25_map: dict[str, float] = {}
    bm25_index, bm25_doc_ids = _get_bm25()
    if bm25_index is not None:
        q_tokens = _tokenize(query)
        scores = bm25_index.get_scores(q_tokens)
        # Normalise BM25 scores to [0, 1]
        max_s = max(scores) if scores.max() > 0 else 1.0
        for doc_id, raw_score in zip(bm25_doc_ids, scores):
            if doc_type and not doc_id.startswith(doc_type):
                continue
            bm25_map[doc_id] = float(raw_score) / max_s

    # ── Fuse scores ───────────────────────────────────────────────────
    all_ids = set(dense_map) | set(bm25_map)
    fused: list[tuple[str, float]] = []
    for doc_id in all_ids:
        d_score = dense_map.get(doc_id, 0.0)
        b_score = bm25_map.get(doc_id, 0.0)
        fused_score = (
            settings.DENSE_WEIGHT * d_score + settings.BM25_WEIGHT * b_score
        )
        fused.append((doc_id, fused_score))

    fused.sort(key=lambda x: x[1], reverse=True)

    # Rebuild result list from dense hits (payload already fetched)
    payload_by_id = {h["payload"]["_doc_id"]: h["payload"] for h in dense_hits}

    # For BM25-only hits we need to load from Qdrant payload — use a search
    # with a very high threshold to avoid re-fetching dense hits.
    missing = [doc_id for doc_id, _ in fused[:k] if doc_id not in payload_by_id]
    if missing:
        # Fetch by exact BM25 doc_ids that weren't in dense results
        for doc_id in missing:
            hits = store.search(query_vec, top_k=1, doc_type=None)
            for h in hits:
                if h["payload"].get("_doc_id") == doc_id:
                    payload_by_id[doc_id] = h["payload"]
                    break

    results = []
    for doc_id, score in fused[:k]:
        if doc_id in payload_by_id:
            results.append({"score": score, "payload": payload_by_id[doc_id]})

    return results
