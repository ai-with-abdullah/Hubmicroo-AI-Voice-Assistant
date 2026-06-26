"""Incremental indexer — upsert or delete a single item without full rebuild.

Builds a text representation of each product/page, embeds it, then upserts
into the vector store. Also maintains the BM25 corpus on disk so retrieval.py
can reload it without re-embedding everything.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import get_settings
from .embedder import embed_text
from . import store

logger = logging.getLogger(__name__)
settings = get_settings()

_BM25_CORPUS_FILE = Path(settings.DATA_DIR) / "bm25_corpus.json"


def _product_text(p: dict[str, Any]) -> str:
    """Flatten a product dict into a searchable text blob."""
    parts = [
        p.get("name", ""),
        p.get("description", ""),
        p.get("category", ""),
        f"price {p.get('price', '')} {p.get('currency', 'PKR')}",
        f"sku {p.get('sku', '')}",
        " ".join(p.get("tags", [])),
        f"stock {'available' if p.get('in_stock', True) else 'out of stock'}",
    ]
    return " | ".join(x for x in parts if x.strip())


def _page_text(pg: dict[str, Any]) -> str:
    return f"{pg.get('title', '')} | {pg.get('content', '')}"


def _load_corpus() -> dict[str, str]:
    """Load the BM25 corpus: {doc_id: text}."""
    if _BM25_CORPUS_FILE.exists():
        return json.loads(_BM25_CORPUS_FILE.read_text())
    return {}


def _save_corpus(corpus: dict[str, str]) -> None:
    _BM25_CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BM25_CORPUS_FILE.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))


def index_product(product: dict[str, Any]) -> None:
    """Upsert a single product into the vector store and BM25 corpus."""
    doc_id = f"product:{product['id']}"
    text = _product_text(product)
    vector = embed_text(text)

    payload = {
        "type": "product",
        "id": product["id"],
        "name": product.get("name", ""),
        "description": product.get("description", ""),
        "price": product.get("price", 0),
        "currency": product.get("currency", "PKR"),
        "in_stock": product.get("in_stock", True),
        "image_url": product.get("image_url", ""),
        "buy_url": product.get("buy_url", ""),
        "category": product.get("category", ""),
        "sku": product.get("sku", ""),
        "tags": product.get("tags", []),
        "_text": text,
    }
    store.upsert(doc_id, vector, payload)

    corpus = _load_corpus()
    corpus[doc_id] = text
    _save_corpus(corpus)
    logger.info("Indexed product %s", product["id"])


def delete_product(product_id: str) -> None:
    """Remove a product from both vector store and BM25 corpus."""
    doc_id = f"product:{product_id}"
    store.delete_by_doc_id(doc_id)

    corpus = _load_corpus()
    corpus.pop(doc_id, None)
    _save_corpus(corpus)
    logger.info("Deleted product %s", product_id)


def index_page(page: dict[str, Any]) -> None:
    """Upsert a FAQ/policy page."""
    doc_id = f"page:{page['id']}"
    text = _page_text(page)
    vector = embed_text(text)

    payload = {
        "type": "page",
        "id": page["id"],
        "title": page.get("title", ""),
        "content": page.get("content", ""),
        "_text": text,
    }
    store.upsert(doc_id, vector, payload)

    corpus = _load_corpus()
    corpus[doc_id] = text
    _save_corpus(corpus)
    logger.info("Indexed page %s", page["id"])


def rebuild_all() -> int:
    """Full rebuild — admin only, rare. Returns count of indexed items."""
    products_file = Path(settings.PRODUCTS_FILE)
    pages_file = Path(settings.PAGES_FILE)

    count = 0
    if products_file.exists():
        products = json.loads(products_file.read_text())
        for p in products:
            index_product(p)
            count += 1

    if pages_file.exists():
        pages = json.loads(pages_file.read_text())
        for pg in pages:
            index_page(pg)
            count += 1

    logger.info("Full rebuild complete: %d items indexed", count)
    return count
