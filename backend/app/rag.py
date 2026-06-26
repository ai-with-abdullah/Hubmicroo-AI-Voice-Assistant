"""RAG: answer any customer question grounded ONLY in Hubmicroo's website.

Flow per message (light — no indexing here):
  detect language -> embed question -> retrieve top chunks from the vector store
  (gathered across ALL pages/products) -> LLM writes one grounded answer in the
  customer's language -> attach product cards.

The system prompt forces the model to use only the retrieved context and to say
"I don't know" instead of inventing answers — so it never makes things up.
"""
from . import config, llm, store
from .indexer import VS
from .services.langdetect import detect_language

_SYSTEM = """You are the shopping assistant for the Hubmicroo online store.
STRICT RULES — follow exactly:
- Use ONLY the information in WEBSITE INFO and MATCHING PRODUCTS below.
- You may ONLY mention products that appear in MATCHING PRODUCTS, and you MUST use their EXACT names and EXACT prices. NEVER invent product names, brands, models, or prices. Do not use any product knowledge from outside this list.
- If MATCHING PRODUCTS is empty and the answer is not in WEBSITE INFO, say you don't have that and suggest contacting Hubmicroo support.
- Reply in {language} only.
- Be short and friendly — this is read aloud. Mention at most {spoken} products.
"""


def _products_block(products) -> str:
    if not products:
        return "(none)"
    lines = []
    for p in products[:config.SPOKEN_RESULTS]:
        stock = "in stock" if p["in_stock"] else "out of stock"
        lines.append(
            f"- {p['name']}: {p['price']} {p['currency']}, {stock}."
            f" {p.get('description', '')}".strip()
        )
    return "\n".join(lines)


def _context(chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        src = c["meta"].get("url") or c["meta"].get("type", "")
        blocks.append(f"[{i}] {c['text']}  (source: {src})")
    return "\n".join(blocks)


def answer(message: str, language: str = None) -> dict:
    """Return {answer, language, products, sources}."""
    lang = language if language in config.LANG_NAMES else detect_language(message)

    # 1) Retrieve relevant knowledge from across the whole site.
    try:
        qvec = llm.embed_one(message)
        chunks = VS.search(qvec, k=config.TOP_K, floor=config.RETRIEVE_FLOOR)
    except llm.LLMError as e:
        return {"answer": str(e), "language": lang, "products": [], "sources": [],
                "error": True}

    # 2) Product cards: prefer products surfaced by retrieval, then live search.
    products, seen = [], set()
    for c in chunks:
        if c["meta"].get("type") == "product":
            p = c["meta"]["product"]
            if p["id"] not in seen:
                products.append(p)
                seen.add(p["id"])
    for p in store.search_products(message):
        if p["id"] not in seen:
            products.append(p)
            seen.add(p["id"])
    products = products[:config.VISUAL_RESULTS]

    # 3) Generate the grounded spoken answer.
    # Only give up if we have neither website text nor matching products.
    if not chunks and not products:
        fallback = {
            "en": "Sorry, I couldn't find that on our store. Please contact Hubmicroo support.",
            "ur": "معاف کیجیے، یہ ہمارے اسٹور پر نہیں ملا۔ براہ کرم ہب مائیکرو سپورٹ سے رابطہ کریں۔",
            "ar": "عذرًا، لم أجد ذلك في متجرنا. يرجى التواصل مع دعم هب مايكرو.",
        }
        return {"answer": fallback[lang], "language": lang,
                "products": products, "sources": []}

    system = _SYSTEM.format(language=config.LANG_NAMES[lang],
                            spoken=config.SPOKEN_RESULTS)
    # Real, authoritative product data goes in explicitly so the model can't
    # invent products/prices even when retrieval is weak.
    user = (
        f"WEBSITE INFO:\n{_context(chunks) or '(none)'}\n\n"
        f"MATCHING PRODUCTS (use ONLY these, with exact names and prices):\n"
        f"{_products_block(products)}\n\n"
        f"CUSTOMER QUESTION: {message}"
    )
    try:
        text = llm.chat(system, user)
    except llm.LLMError as e:
        return {"answer": str(e), "language": lang, "products": products,
                "sources": [], "error": True}

    sources = list({c["meta"].get("url") for c in chunks if c["meta"].get("url")})
    return {"answer": text, "language": lang, "products": products, "sources": sources}
