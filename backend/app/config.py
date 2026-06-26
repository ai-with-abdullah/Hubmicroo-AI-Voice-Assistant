"""Central configuration — all tunables in one place, read from environment."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Compute default data dir relative to this file so local dev works without .env
_APP_DIR = Path(__file__).parent          # backend/app/
_DATA_DIR = _APP_DIR.parent / "data"      # backend/data/
_FRONTEND_DIR = _APP_DIR.parent.parent / "frontend"  # project_root/frontend/


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_MODEL: str = "qwen3:4b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.1
    LLM_NUM_PREDICT: int = 200
    LLM_THINK: bool = False          # keep False for fast/non-thinking mode

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DEVICE: str = "cpu"        # "cuda" on GPU hosts

    # ── Vector store (Qdrant) ─────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "hubmicroo"
    # Set QDRANT_LOCAL_PATH to run in embedded mode (no server needed).
    # Useful for Kaggle / CI / local testing without running a Qdrant container.
    # Leave blank to use the URL server mode (default for Docker production).
    QDRANT_LOCAL_PATH: str = ""

    # ── Retrieval ─────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 8
    BM25_WEIGHT: float = 0.3         # weight for sparse BM25 score
    DENSE_WEIGHT: float = 0.7        # weight for dense cosine score
    PRODUCT_FLOOR: float = 0.35      # min fused score to show a product card
    MAX_PRODUCT_CARDS: int = 3

    # ── Semantic cache ────────────────────────────────────────────────────
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_MAX_SIZE: int = 500        # entries kept in memory

    # ── Voice ─────────────────────────────────────────────────────────────
    WHISPER_MODEL: str = "base"      # tiny/base/small/medium
    WHISPER_DEVICE: str = "cpu"
    PIPER_VOICE_EN: str = "en_US-lessac-medium"
    PIPER_VOICE_UR: str = "ur_PK-usman-medium"
    PIPER_VOICE_AR: str = "ar_JO-kareem-low"
    TTS_ENABLED: bool = True

    # ── Admin / auth ──────────────────────────────────────────────────────
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "changeme"
    SECRET_KEY: str = "change-this-secret-key-in-production"
    JWT_EXPIRE_MINUTES: int = 60

    # ── Server ────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"          # comma-separated list or "*"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Paths — defaults resolve relative to source so local dev works ───────
    # Docker overrides these via ENV vars in the Dockerfile / docker-compose
    DATA_DIR: str = str(_DATA_DIR)
    PRODUCTS_FILE: str = str(_DATA_DIR / "products.json")
    PAGES_FILE: str = str(_DATA_DIR / "pages.json")
    FRONTEND_DIR: str = str(_FRONTEND_DIR)
    PIPER_VOICES_DIR: str = str(_APP_DIR.parent.parent / "piper_voices")

    # ── Greetings / fallback messages ─────────────────────────────────────
    GREETING_EN: str = "Hi! I'm the Hubmicroo shopping assistant. How can I help you today?"
    GREETING_UR: str = "Salam! Main Hubmicroo ka shopping assistant hoon. Aap ki kya madad kar sakta hoon?"
    GREETING_AR: str = "مرحباً! أنا مساعد تسوق هبمايكرو. كيف يمكنني مساعدتك؟"
    FALLBACK_EN: str = "I couldn't find what you're looking for. Would you like to contact our support team?"
    FALLBACK_UR: str = "Mujhe yeh nahi mila. Kya aap hamare support team se baat karna chahenge?"
    FALLBACK_AR: str = "لم أتمكن من العثور على ما تبحث عنه. هل تريد التواصل مع فريق الدعم؟"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
