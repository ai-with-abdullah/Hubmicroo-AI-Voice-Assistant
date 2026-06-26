"""Hybrid retrieval: dense (BGE-M3 via Qdrant) + sparse (BM25 via rank-bm25).

Scores are fused with weighted linear combination, then returned ranked.
BM25 corpus is maintained on disk by indexer.py and hot-reloaded on change.
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


def _bm25_corpus_path() -> Path:
    return Path(settings.DATA_DIR) / "bm25_corpus.json"

# ── BM25 index (lazy-loaded, refreshed when corpus file mtime changes) ────────

_bm25_index: Any = None
_bm25_doc_ids: list[str] = []
_bm25_corpus_mtime: float = 0.0


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _get_bm25() -> tuple[Any, list[str]]:
    """Return (BM25Okapi instance, doc_id list), refreshing if corpus changed."""
    global _bm25_index, _bm25_doc_ids, _bm25_corpus_mtime

    if not _bm25_corpus_path().exists():
        return None, []

    mtime = _bm25_corpus_path().stat().st_mtime
    if _bm25_index is not None and mtime == _bm25_corpus_mtime:
        return _bm25_index, _bm25_doc_ids

    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    corpus_map: dict[str, str] = json.loads(_bm25_corpus_path().read_text())
    _bm25_doc_ids = list(corpus_map.keys())
    tokenized = [_tokenize(v) for v in corpus_map.values()]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus_mtime = mtime
    logger.debug("BM25 index refreshed: %d docs", len(_bm25_doc_ids))
    return _bm25_index, _bm25_doc_ids


def has_bm25_matches(query: str) -> bool:
    """Return True if the query has at least one non-zero BM25 score.

    Used by rag.py to detect likely-typo queries (all tokens unknown to BM25)
    without importing private internals.
    """
    bm25, doc_ids = _get_bm25()
    if bm25 is None or not doc_ids:
        return True  # corpus not ready — assume clean
    tokens = _tokenize(query)
    if not tokens:
        return True
    return float(bm25.get_scores(tokens).max()) > 0.0


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
    query_vec: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieve. Returns ranked list of {score, payload} dicts.

    *doc_type* filters to 'product', 'page', or None (all).
    *query_vec* may be supplied by the caller to avoid a duplicate embed call.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    fetch_k = k * 3  # fetch more candidates to allow re-ranking

    if query_vec is None:
        query_vec = embed_text(query)

    # ── Dense retrieval ────────────────────────────────────────────────────
    dense_hits = store.search(query_vec, top_k=fetch_k, doc_type=doc_type)
    dense_map: dict[str, float] = {}
    payload_by_id: dict[str, dict] = {}
    for h in dense_hits:
        doc_id = h["payload"]["_doc_id"]
        dense_map[doc_id] = h["score"]
        payload_by_id[doc_id] = h["payload"]

    # ── BM25 retrieval ─────────────────────────────────────────────────────
    bm25_map: dict[str, float] = {}
    bm25_index, bm25_doc_ids = _get_bm25()
    if bm25_index is not None and bm25_doc_ids:
        q_tokens = _tokenize(query)
        raw_scores = bm25_index.get_scores(q_tokens)
        max_s = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        for doc_id, raw in zip(bm25_doc_ids, raw_scores):
            # Filter by doc_type if requested
            if doc_type and not doc_id.startswith(doc_type):
                continue
            if raw > 0:
                bm25_map[doc_id] = float(raw) / max_s

    # ── Fuse scores ────────────────────────────────────────────────────────
    all_ids = set(dense_map) | set(bm25_map)
    fused: list[tuple[str, float]] = []
    for doc_id in all_ids:
        score = (
            settings.DENSE_WEIGHT * dense_map.get(doc_id, 0.0)
            + settings.BM25_WEIGHT * bm25_map.get(doc_id, 0.0)
        )
        fused.append((doc_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:k]

    # ── Fetch payloads for BM25-only hits not in dense results ────────────
    missing = [doc_id for doc_id, _ in top if doc_id not in payload_by_id]
    for doc_id in missing:
        payload = store.retrieve_by_doc_id(doc_id)
        if payload:
            payload_by_id[doc_id] = payload
            logger.debug("Fetched BM25-only payload for %s", doc_id)

    # ── Build final result list ────────────────────────────────────────────
    results = []
    for doc_id, score in top:
        if doc_id in payload_by_id:
            results.append({"score": score, "payload": payload_by_id[doc_id]})

    return results
