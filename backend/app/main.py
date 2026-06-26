"""FastAPI app: the orchestration layer.

Endpoints
  GET  /                  -> demo storefront with the voice widget
  GET  /admin             -> admin panel (add product, re-index, stats)
  POST /api/chat          -> grounded answer (voice/text) + product cards
  POST /api/products      -> add a product, then index just that product
  POST /api/reindex       -> rebuild the whole index (rare)
  GET  /api/stats         -> counts
  GET  /api/health        -> Ollama + index status

Voice (speech-to-text and text-to-speech) runs in the browser, so the backend
stays light. The frontend can later be switched to private self-hosted Whisper.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, indexer, llm, rag, store

app = FastAPI(title="Hubmicroo Voice Assistant", version="1.0.0")

# Allow the widget to be embedded on the real Hubmicroo website.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str
    language: str | None = None      # "en" | "ur" | "ar" | None (auto-detect)


class ProductIn(BaseModel):
    id: str
    name: str
    price: float
    currency: str = config.DEFAULT_CURRENCY
    in_stock: bool = True
    image: str = ""
    url: str = ""
    description: str = ""
    category: str = ""
    brand: str = ""
    name_ur: str = ""
    name_ar: str = ""
    features: list[str] = []


@app.post("/api/chat")
def chat(body: ChatIn):
    if not body.message.strip():
        raise HTTPException(400, "Empty message")
    return rag.answer(body.message.strip(), body.language)


@app.post("/api/products")
def add_product(body: ProductIn):
    """Add a product and incrementally index it (no full rebuild)."""
    p = store.add_local_product(body.model_dump())
    try:
        indexer.index_product(p)
    except llm.LLMError as e:
        raise HTTPException(503, str(e))
    return {"added": p["id"], "indexed": True, "chunks": len(indexer.VS)}


@app.post("/api/reindex")
def reindex():
    try:
        return indexer.reindex_all()
    except llm.LLMError as e:
        raise HTTPException(503, str(e))


@app.get("/api/stats")
def stats():
    return {"chunks_indexed": len(indexer.VS),
            "products": len(store.get_all_products())}


@app.get("/api/health")
def health():
    h = llm.health()
    h["chunks_indexed"] = len(indexer.VS)
    return h


# ---- Serve the frontend (demo storefront + widget + admin) -------------
@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/admin")
def admin():
    return FileResponse(config.FRONTEND_DIR / "admin.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
