"""Build / update the RAG index.

IMPORTANT (matches the agreed design): indexing runs only when content changes
— a product is added/edited, or the admin clicks "Re-index". It does NOT run on
every user message. At query time we only *retrieve*, which is cheap.

It turns products + website pages into embedded text chunks in the vector store.
"""
import json

from . import config, llm, store
from .vectorstore import VectorStore

# One shared in-memory store, loaded from disk.
VS = VectorStore()


def _chunk(text: str):
    text = " ".join(text.split())
    if not text:
        return []
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def _product_document(p: dict) -> str:
    stock = "in stock" if p["in_stock"] else "out of stock"
    names = " / ".join(x for x in [p["name"], p.get("name_ur"), p.get("name_ar")] if x)
    feats = ", ".join(p.get("features", []))
    return (
        f"Product: {names}. Category: {p.get('category')}. Brand: {p.get('brand')}. "
        f"Price: {p.get('price')} {p.get('currency')}. Availability: {stock}. "
        f"Features: {feats}. {p.get('description', '')} Link: {p.get('url')}"
    )


def index_product(p: dict):
    """Incrementally (re)index a single product."""
    VS.clear_source(p["id"])
    doc = _product_document(p)
    embeddings = llm.embed([doc])
    VS.add([{
        "text": doc,
        "source_id": p["id"],
        "meta": {"type": "product", "product": p},
    }], embeddings)
    VS.save()


def index_pages(pages):
    """pages: list of {url, title, text} — policies, FAQ, about, shipping ..."""
    for page in pages:
        VS.clear_source(page["url"])
        chunks = _chunk(page.get("text", ""))
        if not chunks:
            continue
        embeddings = llm.embed(chunks)
        VS.add([{
            "text": c,
            "source_id": page["url"],
            "meta": {"type": "page", "title": page.get("title", ""), "url": page["url"]},
        } for c in chunks], embeddings)
    VS.save()


def _load_pages():
    if config.PAGES_FILE.exists():
        return json.loads(config.PAGES_FILE.read_text(encoding="utf-8"))
    return []


def reindex_all() -> dict:
    """Full rebuild: every product + every page. Run on big content changes."""
    VS.clear_all()
    products = store.get_all_products()
    if products:
        docs = [_product_document(p) for p in products]
        embeddings = llm.embed(docs)
        VS.add([{
            "text": docs[i],
            "source_id": products[i]["id"],
            "meta": {"type": "product", "product": products[i]},
        } for i in range(len(products))], embeddings)
    index_pages(_load_pages())
    VS.save()
    return {"products": len(products), "chunks": len(VS)}
