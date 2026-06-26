# Hubmicroo — Self-Hosted AI Voice Assistant

A voice + text shopping assistant for the Hubmicroo store. The customer **speaks
or types** a question in **English, Urdu, or Arabic**; the assistant **answers in
natural language grounded only in Hubmicroo's website** (products, prices, stock,
shipping, returns…) and shows **product cards** with image, price and buy link.

Runs **entirely on your own server**. **No third-party API keys. No monthly
per-query fees.** The "brain" is an open LLM served locally by Ollama.

---

## Why this design (the important part)

| Concern | How it's handled |
|---|---|
| "Answer like Claude, but only about our site" | **RAG**: retrieve relevant text from across all pages → LLM writes one grounded answer. Says "I don't know" instead of inventing. |
| "We don't want to train every time" | ✅ You don't. Adding a product just **re-indexes that one item** (seconds). The LLM is pretrained — never retrained at runtime. |
| "Answer is spread across different pages" | RAG retrieves chunks from **all pages at once** before answering — exactly what a keyword bot can't do. |
| "No monthly API cost / data privacy" | Self-hosted Ollama. No data leaves the server. |
| Live product data | Fetched **live** from your website (or demo catalogue), with image + link + price + stock. |

```
INDEX-TIME (only when products/content change):
   products + website pages → chunks → embeddings → local vector store

QUERY-TIME (every message — light, no indexing):
   voice/text → retrieve relevant chunks → self-hosted LLM → grounded answer
              → speak it (browser TTS) + show product cards
```

---

## Quick start (local demo, ~10 min)

1. **Install Ollama** → https://ollama.com/download  (then it runs in the background)
2. **Run setup** (creates venv, pulls models, builds the first index):
   ```bash
   cd Hubmicroo_VoiceAssistant
   bash scripts/setup.sh
   ```
3. **Start the server:**
   ```bash
   source .venv/bin/activate && cd backend
   uvicorn app.main:app --reload
   ```
4. Open **http://localhost:8000** → click the 🎤 button → ask:
   - *"Do you have wireless headphones under 5000?"*
   - *"کیا آپ کے پاس بلوٹوتھ اسپیکر ہے؟"*
   - *"ما هي سياسة الإرجاع؟"* (return policy)

> Voice in/out uses the browser (Chrome/Edge recommended). Mic needs HTTPS in
> production; `localhost` is exempt.

### Or run with Docker (one command)
```bash
docker compose up --build
# first time, pull models inside the ollama container:
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull bge-m3
docker compose exec backend python -c "from app import indexer; print(indexer.reindex_all())"
```

---

## Connect it to the real Hubmicroo website

Edit `.env` (copy from `.env.example`):

- **Live catalogue:** set `STORE_CATALOGUE_URL` to an endpoint returning your
  products as JSON, **or** `STORE_SEARCH_URL` to a `?q=` search endpoint.
- **Different field names?** Map them: `FIELD_NAME`, `FIELD_PRICE`, `FIELD_IMAGE`…
- **Website text (policies, FAQ):** add pages to `backend/data/pages.json`, then
  click **Re-index** in `/admin`.

**Embed the widget on hubmicroo.com** — host `assistant.js` + `assistant.css`
and add to every page:
```html
<link rel="stylesheet" href="https://YOUR_SERVER/static/assistant.css">
<script>window.HUBMICROO_API_BASE = "https://YOUR_SERVER";</script>
<script src="https://YOUR_SERVER/static/assistant.js"></script>
```

---

## Admin panel — `/admin`
- See chunks indexed, product count, LLM status.
- **Add a product** → it's indexed instantly (no full rebuild, no training).
- **Re-index everything** → after big catalogue/policy changes.

(For production, put `/admin` and the admin APIs behind a login — see "Hardening".)

---

## Server sizing (give this to the client)

| Tier | GPU | Model | Notes |
|---|---|---|---|
| Recommended | 16–24GB VRAM (RTX 4090 / L4) | `qwen2.5:7b` | best answers, <3s voice |
| Budget | 8–12GB | `qwen2.5:3b` (default) | good, lighter |
| Minimum | CPU only | `qwen2.5:3b` | works, slower than 3s |

Change the model in one place: `LLM_MODEL` in `.env`. **Kaggle cannot host the
live server** (sessions expire) — it's only for the optional fine-tune below.

---

## Phase 2 (optional, paid upgrade): fine-tune on Hubmicroo data
`kaggle/finetune_llm_qlora.py` fine-tunes the model on Hubmicroo Q&A using
Kaggle's free GPU. Not required to ship — it improves brand tone and lets you
honestly say the model was **trained on the client's data**. After training,
merge the adapter and load it into Ollama with a `Modelfile`.

---

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat` | `{message, language?}` → grounded answer + product cards |
| POST | `/api/products` | add a product, index it incrementally |
| POST | `/api/reindex` | rebuild the whole index |
| GET | `/api/health` | Ollama + index status |
| GET | `/api/stats` | counts |

---

## Hardening before go-live
- Put `/admin`, `/api/products`, `/api/reindex` behind authentication.
- Restrict `CORS allow_origins` to `https://www.hubmicroo.com`.
- Serve over HTTPS (required for the microphone).
- (Optional) swap browser voice for self-hosted Whisper + MMS-TTS for full
  privacy — the `/api/chat` contract stays the same.

---

*Self-hosted · open-source · no API keys · EN / UR / AR · voice + text.*
