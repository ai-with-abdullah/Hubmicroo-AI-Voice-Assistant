"""Qdrant vector store wrapper — upsert, delete-by-id, search, retrieve.

Uses qdrant-client ≥1.7 API (query_points, retrieve).
client.search() was removed in qdrant-client 1.14+ — do NOT use it.
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models as qm

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    """Return a singleton QdrantClient.

    Modes (controlled by config):
      - QDRANT_LOCAL_PATH set → embedded/local mode (no server; ideal for Kaggle/CI)
      - QDRANT_LOCAL_PATH empty → URL mode (connects to running Qdrant server)
    """
    if settings.QDRANT_LOCAL_PATH:
        logger.info("Qdrant: embedded mode at %s", settings.QDRANT_LOCAL_PATH)
        return QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    logger.info("Qdrant: server mode at %s", settings.QDRANT_URL)
    return QdrantClient(url=settings.QDRANT_URL, timeout=30)


def _point_id(doc_id: str) -> str:
    """Convert a string doc_id to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


def ensure_collection(dim: int = 1024) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    client = _client()
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION)


def upsert(doc_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    """Insert or update a single document."""
    client = _client()
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            qm.PointStruct(
                id=_point_id(doc_id),
                vector=vector,
                payload={**payload, "_doc_id": doc_id},
            )
        ],
    )


def delete_by_doc_id(doc_id: str) -> None:
    """Remove a document by its string ID."""
    client = _client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=qm.PointIdsList(points=[_point_id(doc_id)]),
    )


def search(
    vector: list[float], top_k: int = 5, doc_type: str | None = None
) -> list[dict[str, Any]]:
    """Dense vector search. Returns list of {score, payload} dicts."""
    client = _client()

    query_filter: qm.Filter | None = None
    if doc_type:
        query_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="type",
                    match=qm.MatchValue(value=doc_type),
                )
            ]
        )

    result = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [{"score": p.score, "payload": p.payload} for p in result.points]


def retrieve_by_doc_id(doc_id: str) -> dict[str, Any] | None:
    """Fetch a document's payload by its string ID (exact lookup, no vector needed)."""
    client = _client()
    points = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[_point_id(doc_id)],
        with_payload=True,
        with_vectors=False,
    )
    return points[0].payload if points else None


def collection_stats() -> dict[str, Any]:
    client = _client()
    info = client.get_collection(settings.QDRANT_COLLECTION)
    vectors_cfg = info.config.params.vectors
    return {
        "points_count": info.points_count,
        "status": str(info.status),
        "vector_size": vectors_cfg.size if hasattr(vectors_cfg, "size") else "n/a",
    }
