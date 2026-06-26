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
- If MATCHING PRODUCTS is empty, do NOT mention, list, or suggest any products. Just answer the question from WEBSITE INFO, or chat briefly. Never volunteer a product list the customer did not ask for.
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


# Greetings / small talk in EN/UR/AR (script + romanized). NOT product searches.
_GREETINGS = (
    # English
    "hello", "hi ", "hey", "how are you", "good morning", "good evening",
    "good afternoon", "thanks", "thank you", "whats up", "what's up",
    "who are you", "good night",
    # Romanized Urdu (Latin letters — what people often type)
    "salam", "assalam", "asalam", "aoa", "kaise ho", "kaise hain", "kaise hai",
    "kase ho", "kese ho", "kasy ho", "kya haal", "kya hal", "kia haal", "shukria",
    "shukriya", "theek ho", "kaisa hai", "ap kaise", "aap kaise",
    # Romanized Arabic
    "kaif halak", "kaif halik", "shukran", "marhaba", "ahlan", "sabah",
    # Urdu script
    "السلام علیکم", "اسلام علیکم", "سلام", "آپ کیسے ہیں", "کیسے ہو", "کیسے ہیں",
    "ہیلو", "شکریہ", "کیا حال",
    # Arabic script
    "السلام عليكم", "مرحبا", "اهلا", "أهلا", "كيف حالك", "كيف الحال", "شكرا",
    "صباح الخير", "مساء الخير",
)
_GREETING_REPLY = {
    "en": "Hello! 👋 How can I help you with Hubmicroo products today?",
    "ur": "السلام علیکم! میں ہب مائیکرو کی پروڈکٹس میں آپ کی کیسے مدد کر سکتا ہوں؟",
    "ar": "مرحبًا! كيف يمكنني مساعدتك في منتجات هب مايكرو اليوم؟",
}


def _is_greeting(message: str) -> bool:
    m = f" {message.strip().lower()} "
    if len(message.strip()) > 40:        # long messages are real questions
        return False
    return any(g in m for g in _GREETINGS)


def answer(message: str, language: str = None) -> dict:
    """Return {answer, language, products, sources}."""
    lang = language if language in config.LANG_NAMES else detect_language(message)

    # 0) Greetings / small talk -> friendly reply, NO product search.
    if _is_greeting(message):
        return {"answer": _GREETING_REPLY[lang], "language": lang,
                "products": [], "sources": []}

    # 1) Retrieve relevant knowledge from across the whole site.
    try:
        qvec = llm.embed_one(message)
        chunks = VS.search(qvec, k=config.TOP_K, floor=config.RETRIEVE_FLOOR)
    except llm.LLMError as e:
        return {"answer": str(e), "language": lang, "products": [], "sources": [],
                "error": True}

    # Website text (policies/FAQ/about) is kept separate from products, so the
    # model only talks products when there are MATCHING PRODUCTS.
    page_chunks = [c for c in chunks if c["meta"].get("type") != "product"]

    # 2) Product cards — only when genuinely relevant:
    #    a) product chunks retrieved with a strong score, or
    #    b) a strong fuzzy name/category match. Avoids dumping products on
    #       unrelated questions ("return policy", "where are you", etc.).
    products, seen = [], set()
    for c in chunks:
        if (c["meta"].get("type") == "product"
                and c.get("score", 0) >= config.PRODUCT_FLOOR):
            p = c["meta"]["product"]
            if p["id"] not in seen:
                products.append(p)
                seen.add(p["id"])
    for p in store.search_products(message):     # uses raised MATCH_FLOOR
        if p["id"] not in seen:
            products.append(p)
            seen.add(p["id"])
    products = products[:config.VISUAL_RESULTS]

    # 3) Generate the grounded spoken answer.
    # Give up only if we have neither website text nor matching products.
    if not page_chunks and not products:
        fallback = {
            "en": "Sorry, I couldn't find that on our store. Please contact Hubmicroo support.",
            "ur": "معاف کیجیے، یہ ہمارے اسٹور پر نہیں ملا۔ براہ کرم ہب مائیکرو سپورٹ سے رابطہ کریں۔",
            "ar": "عذرًا، لم أجد ذلك في متجرنا. يرجى التواصل مع دعم هب مايكرو.",
        }
        return {"answer": fallback[lang], "language": lang,
                "products": products, "sources": []}

    system = _SYSTEM.format(language=config.LANG_NAMES[lang],
                            spoken=config.SPOKEN_RESULTS)
    # Products are passed only via MATCHING PRODUCTS, so the model can't invent
    # them and won't volunteer products when the list is empty.
    user = (
        f"WEBSITE INFO:\n{_context(page_chunks) or '(none)'}\n\n"
        f"MATCHING PRODUCTS (use ONLY these, with exact names and prices; "
        f"if empty, do NOT mention any products):\n{_products_block(products)}\n\n"
        f"CUSTOMER QUESTION: {message}"
    )
    try:
        text = llm.chat(system, user)
    except llm.LLMError as e:
        return {"answer": str(e), "language": lang, "products": products,
                "sources": [], "error": True}

    sources = list({c["meta"].get("url") for c in page_chunks if c["meta"].get("url")})
    return {"answer": text, "language": lang, "products": products, "sources": sources}
