# Hubmicroo Voice & Text Shopping Assistant

A **fully self-hosted** AI shopping assistant for a single e-commerce store.  
Customers speak or type in **English, Urdu, or Arabic** — all AI runs on your own server.

---

## Features

- 🎙 **Voice in, voice out** — Whisper STT + Piper TTS, no browser Web Speech API
- 🔍 **Hybrid retrieval** — BGE-M3 dense + BM25 sparse, fused scores
- 🧠 **RAG-grounded answers** — LLM reads retrieved store data; never invents products
- ⚡ **Router + semantic cache** — skips LLM for simple lookups, caches repeated questions
- 🌐 **EN / UR / AR** — Unicode detection + romanised Urdu + code-switching support
- 🗃 **Incremental indexing** — add/edit/delete one product in seconds, no rebuild
- 📦 **Inline product cards** — only matched products, rendered inside the chat message
- 🔒 **Admin panel** — JWT-protected, product CRUD, FAQ editor, query analytics
- 📊 **Eval harness** — 50 test cases across all three languages; swap models and re-run

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- 8 GB RAM minimum (16 GB recommended for GPU)
- Internet access (to pull Ollama model and BGE-M3 on first boot)

### 1. Clone and configure

```bash
git clone https://github.com/ai-with-abdullah/Hubmicroo_VoiceAssistant.git
cd Hubmicroo_VoiceAssistant
cp .env.example .env
# Edit .env — at minimum change ADMIN_PASSWORD and SECRET_KEY
```

### 2. Start all services

```bash
docker compose up -d
```

First boot downloads ~4 GB (Ollama model + BGE-M3). Watch progress:

```bash
docker compose logs -f backend
```

### 3. Open the assistant

| URL | What |
|-----|------|
| `http://localhost:8000/` | Store widget demo |
| `http://localhost:8000/admin` | Admin panel (default: admin / changeme) |
| `http://localhost:8000/docs` | Interactive API docs |
| `http://localhost:8000/api/health` | Health check |

---

## Embedding the Widget

Copy two files to your store's frontend and add to any page:

```html
<link rel="stylesheet" href="https://your-server.com/static/assistant.css">
<script>
  window.HM_API_BASE = "https://your-server.com";
</script>
<script src="https://your-server.com/static/assistant.js"></script>
```

The floating chat button appears automatically. No other changes needed.

---

## Connecting Your Live Catalogue

### Option A — Webhook on product save (recommended)

In your CMS/shop platform, call this endpoint whenever a product changes:

```bash
# Add or update
curl -X POST https://your-server.com/api/products \
  -H "Content-Type: application/json" \
  -d '{"id":"p123","name":"Product Name","price":1999,"in_stock":true,...}'

# Delete
curl -X DELETE https://your-server.com/api/products/p123
```

The product is searchable within 2–3 seconds. No training, no rebuild.

### Option B — Bulk sync script

Edit `backend/data/products.json` then hit:

```bash
curl -X POST http://localhost:8000/api/reindex
```

---

## Swapping the LLM

Change `LLM_MODEL` in `.env` and restart:

```bash
LLM_MODEL=llama3.2:3b   # smaller, faster
LLM_MODEL=phi3:mini      # very fast on CPU
LLM_MODEL=qwen3:8b       # larger, better Urdu
```

Then run the eval harness to compare:

```bash
pip install httpx
python eval/run_eval.py --base-url http://localhost:8000 --out eval/results.json
```

---

## Running the Eval Harness

```bash
# Against local Docker stack
python eval/run_eval.py

# Against a remote server
python eval/run_eval.py --base-url https://your-server.com

# Save full results
python eval/run_eval.py --out eval/results.json
```

Reports: correct-product rate, retrieval hit rate, per-language accuracy, per-type accuracy.  
Exit code `0` = ≥ 70 % correct. Exit code `1` = below threshold (good for CI).

---

## Testing on Kaggle (Free GPU)

See [`kaggle/KAGGLE.md`](kaggle/KAGGLE.md) for instructions.  
Upload [`kaggle/run_on_kaggle.ipynb`](kaggle/run_on_kaggle.ipynb) — 10 cells, runs end to end.

---

## Server Sizing

| Load | RAM | GPU | Recommended |
|------|-----|-----|-------------|
| Demo / testing | 8 GB | None | Any VPS, Kaggle free tier |
| Small store (< 500 products) | 8 GB | Optional | 2-vCPU VPS |
| Medium store (< 5 000 products) | 16 GB | T4 | GPU VPS |
| High traffic | 32 GB | A10 | Dedicated server |

**Scale-to-zero:** Ollama is configured with `OLLAMA_KEEP_ALIVE=5m` — the model unloads from GPU memory after 5 minutes of inactivity, freeing VRAM.

---

## API Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/chat` | None | `{message, language?}` → answer + product cards |
| POST | `/api/voice` | None | Audio file → transcript → answer + optional TTS audio |
| POST | `/api/products` | None | Upsert product, index immediately |
| DELETE | `/api/products/{id}` | None | Remove product from index |
| POST | `/api/reindex` | None | Full rebuild from data files |
| GET | `/api/health` | None | Service health status |
| GET | `/api/stats` | None | Index + cache counts |
| POST | `/admin/login` | — | Get JWT token |
| POST | `/admin/products` | JWT | Admin product upsert |
| DELETE | `/admin/products/{id}` | JWT | Admin product delete |
| POST | `/admin/pages` | JWT | Upsert FAQ/policy page |
| POST | `/admin/reindex` | JWT | Full rebuild (admin) |
| GET | `/admin/stats` | JWT | Extended stats |
| GET | `/admin/queries` | JWT | Recent anonymised queries |

---

## Architecture

```
INDEX-TIME (per product change):
  product edit → embed (BGE-M3) → upsert/delete in Qdrant + BM25 corpus

QUERY-TIME:
  voice/text in
    → Whisper STT (if voice)
    → language detect (EN/UR/AR)
    → router: greeting | lookup | conversational
    → query rewrite (condense to search intent)
    → semantic cache check → serve if hit
    → hybrid retrieve: Qdrant dense + BM25 sparse → fused score
    → route: skip LLM for lookup, call Ollama for conversational
    → grounded answer (retrieval context only)
    → filter product cards (score ≥ PRODUCT_FLOOR)
    → Piper TTS (if voice request)
    → return {answer, products[], lang, cached}
```

---

## Project Structure

```
backend/
  app/
    main.py        FastAPI routes
    config.py      All settings (one place to change anything)
    embedder.py    BGE-M3 singleton
    indexer.py     Incremental upsert/delete
    retrieval.py   Hybrid BM25 + dense retrieval
    router.py      greeting | lookup | conversational classifier
    cache.py       In-process semantic cache
    rag.py         Full query pipeline
    llm.py         Ollama wrapper
    voice.py       Whisper STT + Piper TTS
    langdetect.py  EN/UR/AR detector
    admin.py       Admin endpoints + JWT auth
    store.py       Qdrant wrapper
  data/
    products.json
    pages.json
frontend/
  index.html       Demo store page
  assistant.js     Embeddable widget
  assistant.css    Widget styles (RTL support)
  admin.html       Admin panel SPA
eval/
  test_set.json    50 test cases (EN/UR/AR)
  run_eval.py      Eval runner
kaggle/
  run_on_kaggle.ipynb
  KAGGLE.md
Dockerfile
docker-compose.yml
.env.example
requirements.txt
```
