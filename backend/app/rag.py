"""Core RAG pipeline — query-rewrite → retrieve → route → answer.

Returns a structured response with the answer text and matched product cards.
Product cards appear only when products genuinely match the user's intent.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .config import get_settings
from .embedder import embed_text
from .retrieval import retrieve
from .router import classify
from .cache import get_cache
from .llm import generate
from .langdetect import detect_language

logger = logging.getLogger(__name__)
settings = get_settings()

# ── System prompt (grounding contract) ────────────────────────────────────────
_SYSTEM = (
    "You are the Hubmicroo shopping assistant. "
    "Answer ONLY using the CONTEXT provided below. "
    "Never invent product names, prices, or policies. "
    "If the answer is not in the context, say you don't know and offer to connect "
    "the user with support. "
    "Be concise — 2-4 sentences maximum. "
    "Reply in the same language as the user's question."
)

# ── Query rewrite prompt ───────────────────────────────────────────────────────
_REWRITE_PROMPT = (
    "Correct any spelling mistakes, translate non-English text to English if needed, "
    "and condense the customer message into a short English search query "
    "(max 8 words, keep product names/SKUs exact). "
    "Output ONLY the search query, nothing else.\n\nMessage: {msg}"
)

# Known out-of-catalogue items — when detected, suppress product cards so the
# LLM can tell the user we don't carry them without showing a false match.
_OOB_SIGNALS = re.compile(
    r'\b(iphone|ipad|macbook|airpod|'
    r'samsung\s+(phone|galaxy)|android\s+phone|'
    r'gaming\s+laptop|windows\s+laptop|desktop\s+pc|'
    r'playstation|ps[45]|xbox|nintendo)\b',
    re.IGNORECASE,
)


def _rewrite_query(message: str) -> str:
    """Spell-correct and condense query to a tight English search intent.

    Skips the LLM for short (≤8 word) pure-ASCII queries — BGE-M3 handles minor
    typos and keyword queries well enough without the extra Ollama round-trip.
    Non-ASCII (Arabic / Urdu script) queries are always translated to English so
    that BM25 (English-only corpus) can contribute a score.
    """
    words = message.split()
    # Fast path: short ASCII query — no LLM needed
    if len(words) <= 8 and message.isascii():
        return message
    try:
        raw = generate(_REWRITE_PROMPT.format(msg=message)).strip()
        if not raw:
            return message
        # Take the last non-empty line; models sometimes prepend a preamble
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        result = lines[-1] if lines else message
        # Guard: if the model ignored the instruction and returned something huge, fall back
        return result if result and len(result.split()) <= 20 else message
    except Exception:
        return message


def _build_context(hits: list[dict[str, Any]]) -> str:
    parts = []
    for h in hits:
        p = h["payload"]
        if p.get("type") == "product":
            parts.append(
                f"Product: {p['name']} | Price: {p['price']} {p['currency']} "
                f"| Stock: {'In Stock' if p['in_stock'] else 'Out of Stock'} "
                f"| SKU: {p.get('sku','')} | {p.get('description','')}"
            )
        else:
            parts.append(f"Policy/FAQ: {p.get('title','')}\n{p.get('content','')}")
    return "\n\n".join(parts)


def _extract_product_cards(
    hits: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    cards = []
    for h in hits:
        p = h["payload"]
        if p.get("type") == "product" and h["score"] >= threshold:
            cards.append({
                "id": p["id"],
                "name": p["name"],
                "price": p["price"],
                "currency": p.get("currency", "PKR"),
                "in_stock": p["in_stock"],
                "image_url": p.get("image_url", ""),
                "buy_url": p.get("buy_url", ""),
                "category": p.get("category", ""),
            })
            if len(cards) >= settings.MAX_PRODUCT_CARDS:
                break
    return cards


def _greeting_response(lang: str) -> dict[str, Any]:
    greetings = {
        "en": settings.GREETING_EN,
        "ur": settings.GREETING_UR,
        "ar": settings.GREETING_AR,
    }
    return {"answer": greetings.get(lang, settings.GREETING_EN), "products": [], "lang": lang, "cached": False}


def _fallback_response(lang: str) -> dict[str, Any]:
    fallbacks = {
        "en": settings.FALLBACK_EN,
        "ur": settings.FALLBACK_UR,
        "ar": settings.FALLBACK_AR,
    }
    return {"answer": fallbacks.get(lang, settings.FALLBACK_EN), "products": [], "lang": lang, "cached": False}


def answer(message: str, language: str | None = None) -> dict[str, Any]:
    """Full pipeline: text in → grounded answer + product cards out.

    Returns:
        {answer: str, products: list[dict], lang: str, cached: bool}
    """
    lang = language or detect_language(message)
    msg_type = classify(message)

    if msg_type == "greeting":
        return _greeting_response(lang)

    # ── Query rewrite ──────────────────────────────────────────────────────
    search_query = _rewrite_query(message)

    # ── Semantic cache check ───────────────────────────────────────────────
    cache = get_cache()
    query_vec = embed_text(search_query)
    cached = cache.get(query_vec)
    if cached:
        logger.debug("Cache hit for query: %s", search_query[:60])
        return {
            "answer": cached.answer,
            "products": cached.products,
            "lang": cached.lang,
            "cached": True,
        }

    # ── Retrieval — reuse the vector already computed for the cache check ─────
    hits = retrieve(search_query, top_k=settings.RETRIEVAL_TOP_K, query_vec=query_vec)
    if not hits:
        return _fallback_response(lang)

    # ── Product cards ──────────────────────────────────────────────────────
    # Conversational queries (comparisons, indirect lookups) use a lower
    # threshold so both referenced products surface even at moderate scores.
    # Out-of-catalogue signals (iphone, gaming laptop…) force zero cards so
    # the LLM can answer "we don't sell X" without returning a false match.
    card_threshold = 0.20 if msg_type == "conversational" else settings.PRODUCT_FLOOR
    if _OOB_SIGNALS.search(message):
        product_cards = []
    else:
        product_cards = _extract_product_cards(hits, threshold=card_threshold)

    # ── Route: skip LLM for pure lookup ────────────────────────────────────
    if msg_type == "lookup" and product_cards:
        # Template answer — no LLM
        names = ", ".join(c["name"] for c in product_cards)
        if lang == "ur":
            ans = f"Ji haan, hamare paas yeh available hai: {names}."
        elif lang == "ar":
            ans = f"نعم، لدينا المنتجات التالية: {names}."
        else:
            ans = f"Here's what I found for you: {names}."
    else:
        # ── LLM grounded answer ────────────────────────────────────────
        context = _build_context(hits)
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {message}\n\n"
            f"ANSWER:"
        )
        try:
            ans = generate(prompt, system=_SYSTEM)
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            return _fallback_response(lang)

    # ── Cache the result ───────────────────────────────────────────────────
    cache.set(query_vec, ans, product_cards, lang)

    return {
        "answer": ans,
        "products": product_cards,
        "lang": lang,
        "cached": False,
    }
