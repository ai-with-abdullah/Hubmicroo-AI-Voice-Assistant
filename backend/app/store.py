"""Live product data from your real website (or the bundled demo catalogue).

This is the "connect to the website to fetch real data" layer. It never trains
anything — it just reads your current products so answers and cards are live.
"""
import json

import requests
from rapidfuzz import fuzz

from . import config


def _headers():
    return {"Authorization": f"Bearer {config.STORE_API_KEY}"} if config.STORE_API_KEY else {}


def _normalize(raw: dict) -> dict:
    """Map a raw product from any store shape into our standard fields."""
    fm = config.FIELD_MAP
    return {
        "id": str(raw.get(fm["id"], "")),
        "name": raw.get(fm["name"], ""),
        "name_ur": raw.get("name_ur", ""),
        "name_ar": raw.get("name_ar", ""),
        "price": raw.get(fm["price"]),
        "currency": raw.get(fm["currency"], config.DEFAULT_CURRENCY),
        "in_stock": bool(raw.get(fm["in_stock"], True)),
        "image": raw.get(fm["image"], ""),
        "url": raw.get(fm["url"], ""),
        "description": raw.get(fm["description"], ""),
        "category": raw.get(fm["category"], ""),
        "brand": raw.get(fm["brand"], ""),
        "features": raw.get("features", []),
    }


def get_all_products() -> list:
    """Full catalogue — from your live website if configured, else demo file."""
    if config.STORE_CATALOGUE_URL:
        r = requests.get(config.STORE_CATALOGUE_URL, headers=_headers(),
                         timeout=config.STORE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        items = data.get("products", data) if isinstance(data, dict) else data
        return [_normalize(p) for p in items]
    raw = json.loads(config.PRODUCTS_FILE.read_text(encoding="utf-8"))
    return [_normalize(p) for p in raw]


def _searchable_text(p: dict) -> str:
    parts = [p["name"], p.get("name_ur", ""), p.get("name_ar", ""),
             p.get("category", ""), p.get("brand", ""),
             " ".join(p.get("features", [])), p.get("description", "")]
    return " ".join(x for x in parts if x)


def search_products(query: str, limit: int = None) -> list:
    """Return product cards matching the query.

    Uses your website's own search endpoint if configured; otherwise does fast
    fuzzy matching over the catalogue (no ML, instant).
    """
    limit = limit or config.VISUAL_RESULTS
    if config.STORE_SEARCH_URL:
        r = requests.get(config.STORE_SEARCH_URL, params={"q": query},
                         headers=_headers(), timeout=config.STORE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        items = data.get("products", data) if isinstance(data, dict) else data
        return [_normalize(p) for p in items][:limit]

    scored = []
    for p in get_all_products():
        score = fuzz.token_set_ratio(query.lower(), _searchable_text(p).lower())
        if score >= config.MATCH_FLOOR:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:limit]]


def add_local_product(product: dict) -> dict:
    """Append a product to the demo catalogue (used by the admin panel)."""
    raw = json.loads(config.PRODUCTS_FILE.read_text(encoding="utf-8"))
    raw.append(product)
    config.PRODUCTS_FILE.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return _normalize(product)
