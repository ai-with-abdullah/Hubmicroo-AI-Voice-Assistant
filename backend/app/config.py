"""Configuration for the Hubmicroo Voice Assistant.

Everything runs on YOUR server. The "brain" is a self-hosted open LLM served by
Ollama, so there are no third-party API keys and no monthly per-query fees.

Two kinds of work happen here:
  * INDEX-TIME (rare): website pages + products are embedded into a local vector
    store. Runs only when products/content change (admin "Re-index" or webhook).
  * QUERY-TIME (every message, light): the question is embedded, relevant chunks
    are retrieved, and the LLM writes a grounded answer. No re-indexing per query.

Override any value with an environment variable of the same name.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent       # backend/
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
PRODUCTS_FILE = DATA_DIR / "products.json"              # demo / local catalogue
PAGES_FILE = DATA_DIR / "pages.json"                    # extra website text (policies, FAQ...)
VECTOR_DIR = DATA_DIR / "vector"                        # persisted index lives here

# ---- Self-hosted models (Ollama) ---------------------------------------
# Start Ollama, then: `ollama pull qwen2.5:3b` and `ollama pull bge-m3`.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# The "brain". qwen2.5:3b runs on modest GPUs and speaks EN/UR/AR.
# Bump to qwen2.5:7b (or llama3.1:8b) on a 16GB+ GPU for better answers.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
# Multilingual embedding model for retrieval (great for EN/UR/AR).
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
# Cap answer length — keeps spoken replies short and fast (important on CPU).
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "10m")   # keep model in RAM between calls

# ---- Connect to your real website (live product data) ------------------
# Leave empty to use the bundled demo catalogue (products.json).
STORE_SEARCH_URL = os.getenv("STORE_SEARCH_URL", "")        # GET ?q=<query> -> products
STORE_CATALOGUE_URL = os.getenv("STORE_CATALOGUE_URL", "")  # GET -> full product list
STORE_API_KEY = os.getenv("STORE_API_KEY", "")
STORE_TIMEOUT = float(os.getenv("STORE_TIMEOUT", "5"))

# Map your website's JSON keys to ours so any store shape works.
FIELD_MAP = {
    "id": os.getenv("FIELD_ID", "id"),
    "name": os.getenv("FIELD_NAME", "name"),
    "price": os.getenv("FIELD_PRICE", "price"),
    "currency": os.getenv("FIELD_CURRENCY", "currency"),
    "in_stock": os.getenv("FIELD_STOCK", "in_stock"),
    "image": os.getenv("FIELD_IMAGE", "image"),
    "url": os.getenv("FIELD_URL", "url"),
    "description": os.getenv("FIELD_DESC", "description"),
    "category": os.getenv("FIELD_CATEGORY", "category"),
    "brand": os.getenv("FIELD_BRAND", "brand"),
}

# ---- RAG / retrieval knobs ---------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))                    # chunks fed to the LLM
RETRIEVE_FLOOR = float(os.getenv("RETRIEVE_FLOOR", "0.30"))  # cosine floor

# ---- Product card display ----------------------------------------------
SPOKEN_RESULTS = int(os.getenv("SPOKEN_RESULTS", "3"))
VISUAL_RESULTS = int(os.getenv("VISUAL_RESULTS", "8"))
MATCH_FLOOR = int(os.getenv("MATCH_FLOOR", "72"))      # 0-100 fuzzy floor (higher = stricter)
PRODUCT_FLOOR = float(os.getenv("PRODUCT_FLOOR", "0.45"))  # cosine floor for showing a product card
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "PKR")

LANG_NAMES = {"en": "English", "ur": "Urdu", "ar": "Arabic"}
