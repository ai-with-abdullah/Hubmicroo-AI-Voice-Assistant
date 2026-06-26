"""Qdrant vector store wrapper — upsert, delete-by-id, search.

Manages a single collection with both dense and payload fields.
BM25 is maintained separately in retrieval.py using rank_bm25 on the raw corpus.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def _client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, timeout=30)


def ensure_collection(dim: int = 1024) -> None:
    """Create collection if it doesn't exist."""
    client = _client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION)


def upsert(doc_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    """Insert or update a single document by its string ID."""
    client = _client()
    # Qdrant needs integer or UUID point IDs — convert string ID to UUID
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            qm.PointStruct(
                id=point_id,
                vector=vector,
                payload={**payload, "_doc_id": doc_id},
            )
        ],
    )


def delete_by_doc_id(doc_id: str) -> None:
    """Delete a document by its original string ID."""
    client = _client()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=qm.PointIdsList(points=[point_id]),
    )


def search(
    vector: list[float], top_k: int = 5, doc_type: str | None = None
) -> list[dict[str, Any]]:
    """Dense search. Returns list of {score, payload} dicts."""
    client = _client()
    filt = None
    if doc_type:
        filt = qm.Filter(
            must=[qm.FieldCondition(key="type", match=qm.MatchValue(value=doc_type))]
        )
    hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        query_filter=filt,
        with_payload=True,
    )
    return [{"score": h.score, "payload": h.payload} for h in hits]


def collection_stats() -> dict[str, Any]:
    client = _client()
    info = client.get_collection(settings.QDRANT_COLLECTION)
    return {
        "points_count": info.points_count,
        "status": info.status.value,
        "vector_size": info.config.params.vectors.size,
    }
