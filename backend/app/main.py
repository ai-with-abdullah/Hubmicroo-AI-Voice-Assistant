"""FastAPI application — chat, voice, product CRUD, health, admin."""
from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_settings
from .admin import router as admin_router, record_query
from .indexer import rebuild_all
from . import store as _store
from .rag import answer
from .voice import transcribe, synthesize
from .langdetect import detect_language
from .router import classify
from .llm import is_ollama_up

logger = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure collection exists; optionally seed index."""
    logger.info("Starting Hubmicroo assistant …")
    try:
        _store.ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as exc:
        logger.warning("Qdrant not available at startup: %s", exc)

    # Seed index on first start if data files exist
    bm25_corpus = os.path.join(settings.DATA_DIR, "bm25_corpus.json")
    if not os.path.exists(bm25_corpus):
        logger.info("No BM25 corpus found — running initial index build …")
        try:
            count = rebuild_all()
            logger.info("Initial index: %d items", count)
        except Exception as exc:
            logger.warning("Initial index failed (Qdrant may be starting): %s", exc)

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Hubmicroo Voice & Text Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(admin_router)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    language: str | None = None


class ChatResponse(BaseModel):
    answer: str
    products: list[dict[str, Any]]
    lang: str
    cached: bool


class ProductRequest(BaseModel):
    id: str
    name: str
    description: str = ""
    price: float = 0.0
    currency: str = "PKR"
    in_stock: bool = True
    image_url: str = ""
    buy_url: str = ""
    category: str = ""
    sku: str = ""
    tags: list[str] = []


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    result = answer(req.message, req.language)
    lang = result["lang"]
    msg_type = classify(req.message)
    record_query(req.message, lang, result["cached"], msg_type)
    return ChatResponse(**result)


# ── Voice endpoint ────────────────────────────────────────────────────────────

@app.post("/api/voice")
async def voice(
    audio: UploadFile = File(...),
    language: str = Form(default=""),
    tts: str = Form(default="true"),
):
    """Receive audio, transcribe via Whisper, run pipeline, return answer + optional audio."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio")

    hint = language if language in ("en", "ur", "ar") else None
    try:
        transcript, detected_lang = transcribe(audio_bytes, hint)
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail="Transcription failed")

    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Could not understand audio")

    result = answer(transcript, detected_lang)
    lang = result["lang"]
    record_query(transcript, lang, result["cached"], classify(transcript))

    response_data: dict[str, Any] = {
        "transcript": transcript,
        "answer": result["answer"],
        "products": result["products"],
        "lang": lang,
        "cached": result["cached"],
        "audio_b64": None,
    }

    if tts.lower() == "true":
        try:
            wav = synthesize(result["answer"], lang)
            if wav:
                response_data["audio_b64"] = base64.b64encode(wav).decode()
        except Exception as exc:
            logger.warning("TTS failed (non-fatal): %s", exc)

    return response_data


# ── Product endpoints (public write protected by admin auth via /admin/products) ──

@app.post("/api/products")
def add_product(product: ProductRequest):
    """Public upsert — used by the store's webhook/integration."""
    from .indexer import index_product
    from .admin import _upsert_product_file

    data = product.model_dump()
    _upsert_product_file(data)
    index_product(data)
    return {"status": "indexed", "id": product.id}


@app.delete("/api/products/{product_id}")
def remove_product(product_id: str):
    from .indexer import delete_product
    from .admin import _delete_product_file

    _delete_product_file(product_id)
    delete_product(product_id)
    return {"status": "deleted", "id": product_id}


@app.post("/api/reindex")
def reindex():
    count = rebuild_all()
    return {"status": "done", "indexed": count}


# ── Health & stats ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    qdrant_ok = True
    try:
        _store.collection_stats()
    except Exception:
        qdrant_ok = False
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "ollama": is_ollama_up(),
    }


@app.get("/api/stats")
def stats():
    try:
        col = _store.collection_stats()
    except Exception:
        col = {}
    from .cache import get_cache
    return {
        "collection": col,
        "cache": get_cache().stats(),
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────
_FRONTEND = "/app/frontend"
if os.path.isdir(_FRONTEND):
    app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(os.path.join(_FRONTEND, "index.html"))

    @app.get("/admin", response_class=HTMLResponse)
    def admin_ui():
        return FileResponse(os.path.join(_FRONTEND, "admin.html"))
