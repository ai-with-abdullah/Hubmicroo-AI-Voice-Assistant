"""Admin router — product CRUD, page CRUD, index ops, recent queries.

All write endpoints require a valid JWT (obtained via /admin/login).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import get_settings
from .indexer import index_product, index_page, delete_product, rebuild_all
from . import store as _store
from .llm import is_ollama_up

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin", tags=["admin"])

# ── Auth ───────────────────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2 = OAuth2PasswordBearer(tokenUrl="/admin/login")
_ALGORITHM = "HS256"


def _make_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": int(time.time()) + settings.JWT_EXPIRE_MINUTES * 60,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=_ALGORITHM)


def _require_admin(token: Annotated[str, Depends(_oauth2)]) -> str:
    try:
        data = jwt.decode(token, settings.SECRET_KEY, algorithms=[_ALGORITHM])
        username: str = data.get("sub", "")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if username != settings.ADMIN_USERNAME:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return username


AdminUser = Annotated[str, Depends(_require_admin)]

# ── Recent queries ring-buffer (anonymised — no PII stored) ───────────────────
_recent_queries: deque[dict[str, Any]] = deque(maxlen=200)


def record_query(query: str, lang: str, cached: bool, msg_type: str) -> None:
    _recent_queries.appendleft({
        "ts": int(time.time()),
        "lang": lang,
        "cached": cached,
        "type": msg_type,
        "chars": len(query),  # length only, not the text
    })


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/login")
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    if form.username != settings.ADMIN_USERNAME or form.password != settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=400, detail="Bad credentials")
    return {"access_token": _make_token(form.username), "token_type": "bearer"}


# ── Products ──────────────────────────────────────────────────────────────────

class ProductIn(BaseModel):
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


@router.post("/products")
def upsert_product(product: ProductIn, _: AdminUser):
    """Add or update a product — indexed immediately, no rebuild needed."""
    data = product.model_dump()
    _upsert_product_file(data)
    index_product(data)
    return {"status": "indexed", "id": product.id}


@router.delete("/products/{product_id}")
def remove_product(product_id: str, _: AdminUser):
    _delete_product_file(product_id)
    delete_product(product_id)
    return {"status": "deleted", "id": product_id}


# ── Pages ─────────────────────────────────────────────────────────────────────

class PageIn(BaseModel):
    id: str
    title: str
    content: str


@router.post("/pages")
def upsert_page(page: PageIn, _: AdminUser):
    data = page.model_dump()
    _upsert_page_file(data)
    index_page(data)
    return {"status": "indexed", "id": page.id}


# ── Index ops ─────────────────────────────────────────────────────────────────

@router.post("/reindex")
def full_reindex(_: AdminUser):
    """Rarely used — full rebuild from data files."""
    count = rebuild_all()
    return {"status": "done", "indexed": count}


@router.get("/stats")
def stats(_: AdminUser):
    try:
        col = _store.collection_stats()
    except Exception:
        col = {"error": "Qdrant unavailable"}
    return {
        "collection": col,
        "ollama_up": is_ollama_up(),
        "recent_queries_buffered": len(_recent_queries),
    }


@router.get("/queries")
def recent_queries(_: AdminUser):
    return list(_recent_queries)


# ── File helpers ──────────────────────────────────────────────────────────────

def _load_products() -> list[dict]:
    p = Path(settings.PRODUCTS_FILE)
    return json.loads(p.read_text()) if p.exists() else []


def _save_products(products: list[dict]) -> None:
    p = Path(settings.PRODUCTS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(products, ensure_ascii=False, indent=2))


def _upsert_product_file(product: dict) -> None:
    products = _load_products()
    products = [x for x in products if x["id"] != product["id"]]
    products.append(product)
    _save_products(products)


def _delete_product_file(product_id: str) -> None:
    products = [x for x in _load_products() if x["id"] != product_id]
    _save_products(products)


def _load_pages() -> list[dict]:
    p = Path(settings.PAGES_FILE)
    return json.loads(p.read_text()) if p.exists() else []


def _save_pages(pages: list[dict]) -> None:
    p = Path(settings.PAGES_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pages, ensure_ascii=False, indent=2))


def _upsert_page_file(page: dict) -> None:
    pages = _load_pages()
    pages = [x for x in pages if x["id"] != page["id"]]
    pages.append(page)
    _save_pages(pages)
